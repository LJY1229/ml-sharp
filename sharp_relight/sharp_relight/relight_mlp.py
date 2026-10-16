"""Small angle-conditioned Gaussian color network; no SHARP dependency here."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class RelightMLP(nn.Module):
    def __init__(self, hidden=64, layers=3, frequencies=4):
        super().__init__()
        self.config = dict(hidden=hidden, layers=layers, frequencies=frequencies)
        self.frequencies = frequencies
        # Normalized XYZ, Fourier XYZ, original linear RGB, sin/cos light angle.
        dims = 3 + 6 * frequencies + 3 + 2
        blocks = []
        for _ in range(layers):
            blocks.extend([nn.Linear(dims, hidden), nn.SiLU()])
            dims = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(dims, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, xyz, rgb, degrees):
        angle = torch.as_tensor(degrees, device=xyz.device, dtype=xyz.dtype)
        angle = angle * (math.pi / 180)
        encoded = [xyz]
        for level in range(self.frequencies):
            phase = xyz * (math.pi * 2 ** level)
            encoded.extend([phase.sin(), phase.cos()])
        direction = torch.stack([angle.sin(), angle.cos()]).expand(*xyz.shape[:-1], 2)
        encoded.extend([rgb, direction])
        delta = self.head(self.body(torch.cat(encoded, dim=-1)))
        # Residual in logit space keeps linear RGB bounded and starts near identity.
        # Clamp is fixed preprocessing of base RGB, not of the predicted residual.
        return torch.sigmoid(torch.logit(rgb.clamp(1e-4, 1 - 1e-4)) + delta)


def normalize_positions(xyz):
    """Robust, scene-specific normalization; parameters need no learned scene ID."""
    flat = xyz.reshape(-1, 3)
    # Subsample only to estimate normalization, never to remove Gaussians.
    sample = flat[::max(1, len(flat) // 32768)]
    center = sample.median(dim=0).values
    scale = (sample - center).norm(dim=-1).median().clamp_min(1e-4)
    return ((xyz - center) / scale).clamp(-10, 10)


def predict_colors(model, xyz, rgb, degrees, chunk_size=65536):
    """Chunk plus activation recomputation limits MLP training activation memory."""
    shape = rgb.shape
    positions = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    outputs = []
    for start in range(0, len(colors), chunk_size):
        x, c = positions[start:start + chunk_size], colors[start:start + chunk_size]
        if model.training and torch.is_grad_enabled():
            outputs.append(checkpoint(model, x, c, degrees, use_reentrant=False))
        else:
            outputs.append(model(x, c, degrees))
    return torch.cat(outputs, dim=0).reshape(shape)
