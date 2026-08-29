"""Policy architecture: BEV CNN + six-dimensional state MLP."""

from __future__ import annotations

import torch
from torch import nn


class BEVScalarPolicy(nn.Module):
    """Return MetaDrive ``[steering, throttle/brake]`` actions in [-1, 1]."""

    def __init__(self, in_channels: int = 1, scalar_dim: int = 6) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scalar_dim = scalar_dim
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, 128), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.action_head = nn.Sequential(nn.Linear(160, 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, bev: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        if bev.ndim != 4 or bev.shape[1] != self.in_channels or bev.shape[-2:] != (64, 64):
            raise ValueError(f"Expected BEV (B,{self.in_channels},64,64), received {tuple(bev.shape)}")
        if scalar.ndim != 2 or scalar.shape[1] != self.scalar_dim:
            raise ValueError(f"Expected scalar (B,{self.scalar_dim}), received {tuple(scalar.shape)}")
        return self.action_head(torch.cat((self.bev_encoder(bev), self.scalar_encoder(scalar)), dim=1))
