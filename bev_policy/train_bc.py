"""Train the clear-condition CNN + MLP behavioural-cloning policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from bev_policy.dataset import BEVDataset, episode_files, split_files
from bev_policy.model import BEVScalarPolicy


def loss_on(model, loader, loss_fn, device):
    model.eval(); total = 0.0
    with torch.no_grad():
        for bev, scalar, action in loader:
            total += float(loss_fn(model(bev.to(device), scalar.to(device)), action.to(device))) * len(action)
    return total / len(loader.dataset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("bev_policy/data/clear"))
    parser.add_argument("--checkpoint", type=Path, default=Path("bev_policy/checkpoints/bc_clear.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    train_files, val_files = split_files(episode_files(args.data_dir))
    train = BEVDataset(train_files); val = BEVDataset(val_files, train.mean, train.std)
    if train.bev.shape[1] != val.bev.shape[1]:
        raise ValueError("BEV channel count differs between train and validation episodes.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVScalarPolicy(train.bev.shape[1]).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss(); train_loader = DataLoader(train, args.batch_size, shuffle=True); val_loader = DataLoader(val, args.batch_size)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True); best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bev, scalar, action in train_loader:
            loss = loss_fn(model(bev.to(device), scalar.to(device)), action.to(device)); optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        train_mse, val_mse = loss_on(model, train_loader, loss_fn, device), loss_on(model, val_loader, loss_fn, device)
        print({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best:
            best, stale = val_mse, 0
            torch.save({"model_state": model.state_dict(), "in_channels": int(train.bev.shape[1]),
                        "scalar_mean": train.mean.tolist(), "scalar_std": train.std.tolist(), "best_val_mse": best,
                        "train_episodes": [p.name for p in train_files], "validation_episodes": [p.name for p in val_files]}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"saved best checkpoint: {args.checkpoint}; validation MSE={best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
