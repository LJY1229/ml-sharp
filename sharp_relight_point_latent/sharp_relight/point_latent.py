"""Single-scene trainable Gaussian features and shared angle decoder."""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PointLatentRelighter(nn.Module):
    def __init__(self, num_points, latent_dim=32, hidden=128, layers=3):
        super().__init__()
        self.config = dict(num_points=num_points, latent_dim=latent_dim,
                           hidden=hidden, layers=layers)
        self.features = nn.Parameter(torch.randn(num_points, latent_dim) * 0.1)
        blocks = []
        width = latent_dim + 2
        for _ in range(layers):
            blocks.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        blocks.append(nn.Linear(width, 3))
        self.decoder = nn.Sequential(*blocks)

    def decode(self, features, angle):
        radians = torch.as_tensor(angle, dtype=features.dtype, device=features.device) * math.pi / 180
        direction = torch.stack([radians.sin(), radians.cos()]).expand(len(features), 2)
        # Direct bounded linear RGB, no original-color logit residual.
        return self.decoder(torch.cat([features, direction], dim=-1)).sigmoid()

    def forward(self, angle, chunk_size=32768):
        parts = []
        for start in range(0, len(self.features), chunk_size):
            features = self.features[start:start + chunk_size]
            if self.training and torch.is_grad_enabled():
                part = checkpoint(self.decode, features, angle, use_reentrant=False)
            else:
                part = self.decode(features, angle)
            parts.append(part)
        return torch.cat(parts, dim=0).unsqueeze(0)
