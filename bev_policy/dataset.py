"""Validated, episode-disjoint BEV behavioural-cloning data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def episode_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("seed_*.npz"))
    if not files:
        raise FileNotFoundError(f"No seed_*.npz episodes in {directory.resolve()}")
    return files


def load_episode(path: Path, with_weather: bool = False):
    with np.load(path, allow_pickle=False) as data:
        if not {"bev", "scalar", "action"} <= set(data.files):
            raise ValueError(f"{path.name} must contain bev, scalar, action.")
        bev = np.asarray(data["bev"])
        bev = bev if bev.dtype == np.uint8 else bev.astype(np.float32)   # v2 BEVs are stored as uint8 {0,1}
        scalar = np.asarray(data["scalar"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        weather = np.asarray(data["weather"], dtype=np.float32) if with_weather else None
    if bev.ndim == 3:
        bev = bev[:, None]
    if bev.ndim != 4 or bev.shape[-2:] != (64, 64) or scalar.shape != (len(bev), 6) or action.shape != (len(bev), 2):
        raise ValueError(f"{path.name}: expected (N,C,64,64), (N,6), (N,2); got {bev.shape}, {scalar.shape}, {action.shape}")
    if len(bev) == 0 or not all(np.isfinite(x).all() for x in (bev, scalar, action)) or np.abs(action).max() > 1.001:
        raise ValueError(f"{path.name}: invalid or out-of-range sample.")
    if with_weather:
        if weather is None or weather.shape != (len(bev), 4):
            raise ValueError(f"{path.name}: missing/invalid weather array")
        return bev, scalar, action, weather
    return bev, scalar, action


def split_files(files: Sequence[Path]) -> tuple[list[Path], list[Path]]:
    if len(files) < 3:
        raise ValueError("Collect at least three seed-disjoint episodes before training.")
    count = max(1, round(len(files) * 0.2))
    return list(files[:-count]), list(files[-count:])


class BEVDataset(Dataset):
    def __init__(self, files: Sequence[Path], mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 with_weather: bool = False) -> None:
        self.with_weather = with_weather
        chunks = [load_episode(path, with_weather) for path in files]
        self.weather = np.concatenate([c[3] for c in chunks]) if with_weather else None
        self.bev = np.concatenate([chunk[0] for chunk in chunks])
        scalar = np.concatenate([chunk[1] for chunk in chunks])
        self.mean = scalar.mean(axis=0) if mean is None else mean.astype(np.float32)
        self.std = np.maximum(scalar.std(axis=0), 1e-3) if std is None else std.astype(np.float32)
        self.scalar = (scalar - self.mean) / self.std
        self.action = np.concatenate([chunk[2] for chunk in chunks])

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, index: int):
        if self.with_weather:
            return (torch.from_numpy(self.bev[index]).float(), torch.from_numpy(self.scalar[index]),
                    torch.from_numpy(self.action[index]), torch.from_numpy(self.weather[index]))
        return torch.from_numpy(self.bev[index]).float(), torch.from_numpy(self.scalar[index]), torch.from_numpy(self.action[index])
