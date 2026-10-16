"""Two scene-conditioned relighters. No trainable per-scene/per-point table."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def flat_to_map(value, grid):
    layers, height, width = grid
    if value.shape[1] != layers * height * width:
        raise ValueError("Gaussian count does not match captured SHARP grid")
    return value.reshape(value.shape[0], layers, height, width, -1).permute(
        0, 1, 4, 2, 3).flatten(1, 2)


def map_to_flat(value, grid):
    layers, height, width = grid
    value = F.interpolate(value, size=(height, width), mode="bilinear", align_corners=False)
    return value.reshape(value.shape[0], layers, 3, height, width).permute(
        0, 1, 3, 4, 2).flatten(1, 3)


class FiLMBlock(nn.Module):
    def __init__(self, dim_in, dim_out, condition_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim_out)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, dim_out)
        self.affine = nn.Linear(condition_dim, dim_out * 2)
        self.skip = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, condition):
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        y = self.norm1(self.conv1(x))
        y = y * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        y = F.silu(y)
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class SceneRelighter(nn.Module):
    def __init__(self, architecture, layers=2, feature_channels=64, width=32):
        super().__init__()
        if architecture not in ("sharp", "unet") or width % 8:
            raise ValueError("architecture must be sharp/unet; width must be a multiple of 8")
        self.config = dict(architecture=architecture, layers=layers,
                           feature_channels=feature_channels, width=width)
        self.architecture = architecture
        self.angle = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64))
        # RGB + normalized XYZ + alpha for each layer, plus source RGB.
        attributes = 7 * layers + 3
        if architecture == "sharp":
            self.project = nn.Conv2d(feature_channels + attributes, width, 1)
            self.blocks = nn.ModuleList([FiLMBlock(width, width) for _ in range(4)])
        else:
            self.enc = nn.ModuleList([
                FiLMBlock(attributes, width), FiLMBlock(width, width * 2),
                FiLMBlock(width * 2, width * 4), FiLMBlock(width * 4, width * 8)])
            self.dec = nn.ModuleList([
                FiLMBlock(width * 12, width * 4), FiLMBlock(width * 6, width * 2),
                FiLMBlock(width * 3, width)])
        self.output = nn.Conv2d(width, 3 * layers, 1)

    def forward(self, attributes, features, degrees, grid):
        angle = torch.as_tensor(degrees, dtype=attributes.dtype, device=attributes.device)
        angle = angle.reshape(-1) * (math.pi / 180)
        condition = self.angle(torch.stack([angle.sin(), angle.cos()], dim=1))
        if self.architecture == "sharp":
            if features is None:
                raise ValueError("SHARP feature head requires captured features")
            features = F.interpolate(features, size=attributes.shape[-2:], mode="bilinear",
                                     align_corners=False)
            x = self.project(torch.cat([features, attributes], dim=1))
            for block in self.blocks:
                x = block(x, condition)
        else:
            skips = []
            x = attributes
            for index, block in enumerate(self.enc):
                if index:
                    x = F.avg_pool2d(x, 2)
                x = block(x, condition)
                skips.append(x)
            for block, skip in zip(self.dec, reversed(skips[:-1])):
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = block(torch.cat([x, skip], dim=1), condition)
        # Both alternatives directly predict linear RGB for a fair comparison.
        # Interpolate logits, then sigmoid; never treat the output as a 2D image.
        return map_to_flat(self.output(x), grid).sigmoid()
