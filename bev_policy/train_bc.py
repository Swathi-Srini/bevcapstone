"""Train the clear-condition CNN + MLP behavioural-cloning policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from bev_policy.dataset import BEVDataset, episode_files, split_files
from bev_policy.model import BEVScalarPolicy

def action_loss(prediction, target, brake_weight: float) -> torch.Tensor:
    """MSE that increases longitudinal error weight on expert braking samples."""

    squared_error = (prediction - target).square()
    if brake_weight > 1.0:
        weights = torch.ones_like(squared_error)
        weights[:, 1] = torch.where(target[:, 1] < 0.0, brake_weight, 1.0)
        squared_error = squared_error * weights
    return squared_error.mean()


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
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional compatible BC checkpoint used to initialise the policy weights.")
    parser.add_argument("--brake-weight", type=float, default=1.0,
                        help="Longitudinal loss multiplier for expert braking samples (must be at least 1).")
    args = parser.parse_args()
    if args.brake_weight < 1.0:
        raise ValueError("--brake-weight must be at least 1.")
    train_files, val_files = split_files(episode_files(args.data_dir))
    train = BEVDataset(train_files); val = BEVDataset(val_files, train.mean, train.std)
    if train.bev.shape[1] != val.bev.shape[1]:
        raise ValueError("BEV channel count differs between train and validation episodes.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVScalarPolicy(train.bev.shape[1]).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        if int(initial["in_channels"]) != train.bev.shape[1]:
            raise ValueError("Initial checkpoint BEV channel count is incompatible with the dataset.")
        model.load_state_dict(initial["model_state"])
        print(f"initialised model weights from {args.init_checkpoint}")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = lambda prediction, target: action_loss(prediction, target, args.brake_weight)
    train_loader = DataLoader(train, args.batch_size, shuffle=True); val_loader = DataLoader(val, args.batch_size)
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
                        "train_episodes": [p.name for p in train_files], "validation_episodes": [p.name for p in val_files],
                        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                        "brake_weight": args.brake_weight}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"saved best checkpoint: {args.checkpoint}; validation MSE={best:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
