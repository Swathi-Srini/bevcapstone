"""Policy architecture: BEV CNN + six-dimensional state MLP."""

from __future__ import annotations

import torch
from torch import nn


class BEVScalarPolicy(nn.Module):
    """Return MetaDrive ``[steering, throttle/brake]`` actions in [-1, 1]."""

    def __init__(self, in_channels: int = 1, scalar_dim: int = 6, weather_dim: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.scalar_dim = scalar_dim
        self.weather_dim = weather_dim
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, 128), nn.ReLU(),
        )
        self.scalar_encoder = nn.Sequential(nn.Linear(scalar_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        # Optional weather branch: explicit [one-hot(clear, fog, rain), intensity] -> 16 (supplied, not detected).
        self.weather_encoder = (nn.Sequential(nn.Linear(weather_dim, 16), nn.ReLU()) if weather_dim else None)
        self.action_head = nn.Sequential(nn.Linear(160 + (16 if weather_dim else 0), 128), nn.ReLU(), nn.Linear(128, 2), nn.Tanh())

    def forward(self, bev: torch.Tensor, scalar: torch.Tensor, weather: torch.Tensor | None = None) -> torch.Tensor:
        if bev.ndim != 4 or bev.shape[1] != self.in_channels or bev.shape[-2:] != (64, 64):
            raise ValueError(f"Expected BEV (B,{self.in_channels},64,64), received {tuple(bev.shape)}")
        if scalar.ndim != 2 or scalar.shape[1] != self.scalar_dim:
            raise ValueError(f"Expected scalar (B,{self.scalar_dim}), received {tuple(scalar.shape)}")
        feats = [self.bev_encoder(bev), self.scalar_encoder(scalar)]
        if self.weather_dim:
            if weather is None or weather.shape[1:] != (self.weather_dim,):
                raise ValueError(f"Expected weather (B,{self.weather_dim})")
            feats.append(self.weather_encoder(weather))
        return self.action_head(torch.cat(feats, dim=1))
