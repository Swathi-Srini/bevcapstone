"""Train the BEV CNN + MLP behavioural-cloning policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bev_policy.dataset import BEVDataset, episode_files, split_files
from bev_policy.model import BEVScalarPolicy


def action_loss(prediction, target, brake_weight: float) -> torch.Tensor:
    """MSE with an optional larger longitudinal loss on braking targets."""
    squared_error = (prediction - target).square()
    if brake_weight > 1.0:
        weights = torch.ones_like(squared_error)
        weights[:, 1] = torch.where(target[:, 1] < -0.05, brake_weight, 1.0)
        squared_error = squared_error * weights
    return squared_error.mean()


def loss_on(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            bev, scalar, action = batch[:3]
            weather = batch[3].to(device) if len(batch) > 3 else None
            prediction = model(bev.to(device), scalar.to(device), weather)
            total += float(loss_fn(prediction, action.to(device))) * len(action)
    return total / len(loader.dataset)


def load_initial_policy(model: BEVScalarPolicy, checkpoint: dict, device: torch.device) -> None:
    """Transfer compatible BC weights while allowing pool/weather branches to differ."""
    source_pool = int(checkpoint.get("bev_pool_size", 1))
    source_weather = int(checkpoint.get("weather_dim", 0))
    initial_model = BEVScalarPolicy(
        in_channels=int(checkpoint["in_channels"]),
        scalar_dim=int(checkpoint.get("scalar_dim", 6)),
        weather_dim=source_weather,
        bev_pool_size=source_pool,
    )
    initial_model.load_state_dict(checkpoint["model_state"])
    source_state = initial_model.state_dict()
    target_state = model.state_dict()
    transferable = {
        name: value for name, value in source_state.items()
        if name in target_state and value.shape == target_state[name].shape
    }
    skipped = set(source_state) - set(transferable)
    allowed_skipped: set[str] = set()
    if source_pool != model.bev_pool_size:
        allowed_skipped.add("bev_encoder.8.weight")
    if source_weather != model.weather_dim:
        allowed_skipped.add("action_head.0.weight")
        if source_weather:
            allowed_skipped.update({"weather_encoder.0.weight", "weather_encoder.0.bias"})
    if skipped != allowed_skipped:
        raise ValueError(f"Initial checkpoint has incompatible policy layers: {sorted(skipped)}")
    model.load_state_dict(transferable, strict=False)
    if skipped:
        print(f"reinitialized incompatible policy layers: {sorted(skipped)}")
    model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("bev_policy/data/clear"))
    parser.add_argument("--checkpoint", type=Path, default=Path("bev_policy/checkpoints/bc_clear.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed for model initialization and shuffled training batches.")
    parser.add_argument("--min-brake-samples", type=int, default=0,
                        help="Require at least this many braking labels in both data splits.")
    parser.add_argument("--bev-pool-size", type=int, default=4,
                        help="Spatial BEV feature-map size; 1 reproduces legacy global pooling.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Optional compatible BC checkpoint used to initialise policy weights.")
    parser.add_argument("--brake-weight", type=float, default=1.0,
                        help="Longitudinal loss multiplier on expert braking samples (at least 1).")
    parser.add_argument("--weather-input", choices=("none", "explicit"), default="none",
                        help="Add the 4-D supplied-weather branch; episode files must contain weather.")
    parser.add_argument("--only-clear", action="store_true",
                        help="Train only on episodes labelled clear in metadata_json.")
    parser.add_argument("--val-count", type=int, default=0,
                        help="Hold out this many episodes, spread across the weather schedule.")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0.0
            or args.patience < 1 or args.min_brake_samples < 0 or args.bev_pool_size < 1
            or args.brake_weight < 1.0 or args.val_count < 0):
        raise ValueError("Training settings must be positive; val-count and min-brake-samples must be non-negative.")

    files = episode_files(args.data_dir)
    if args.only_clear:
        files = [path for path in files
                 if json.loads(str(np.load(path)["metadata_json"]))["weather"][0] == "clear"]
    if args.val_count:
        if len(files) <= args.val_count or args.val_count < 1:
            raise ValueError("--val-count must be smaller than the number of selected episodes.")
        stride = max(1, len(files) // args.val_count)
        val_files = files[::stride][:args.val_count]
        train_files = [path for path in files if path not in val_files]
    else:
        train_files, val_files = split_files(files)

    initial = None
    scalar_mean = scalar_std = None
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        scalar_mean = np.asarray(initial["scalar_mean"], dtype=np.float32)
        scalar_std = np.asarray(initial["scalar_std"], dtype=np.float32)
        if (scalar_mean.shape != (6,) or scalar_std.shape != (6,)
                or not np.isfinite(scalar_mean).all() or not np.isfinite(scalar_std).all()
                or np.any(scalar_std <= 0.0)):
            raise ValueError("Initial checkpoint must contain finite six-dimensional scalar normalization.")

    use_weather = args.weather_input == "explicit"
    train = BEVDataset(train_files, scalar_mean, scalar_std, with_weather=use_weather)
    val = BEVDataset(val_files, train.mean, train.std, with_weather=use_weather)
    if train.bev.shape[1] != val.bev.shape[1]:
        raise ValueError("BEV channel count differs between train and validation episodes.")
    if initial is not None and int(initial["in_channels"]) != train.bev.shape[1]:
        raise ValueError("Initial checkpoint BEV channel count is incompatible with the dataset.")

    train_brake_samples = int(np.count_nonzero(train.action[:, 1] < -0.05))
    val_brake_samples = int(np.count_nonzero(val.action[:, 1] < -0.05))
    print({"train_samples": len(train), "validation_samples": len(val),
           "train_brake_samples": train_brake_samples, "validation_brake_samples": val_brake_samples})
    if train_brake_samples < args.min_brake_samples or val_brake_samples < args.min_brake_samples:
        raise ValueError(
            f"Need at least {args.min_brake_samples} braking samples in each split; "
            f"got {train_brake_samples} and {val_brake_samples}."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weather_dim = 4 if use_weather else 0
    model = BEVScalarPolicy(
        in_channels=train.bev.shape[1], weather_dim=weather_dim,
        bev_pool_size=args.bev_pool_size,
    ).to(device)
    if initial is not None:
        load_initial_policy(model, initial, device)
        print(f"initialised model weights from {args.init_checkpoint}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = lambda prediction, target: action_loss(prediction, target, args.brake_weight)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, args.batch_size)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            bev, scalar, action = batch[:3]
            weather = batch[3].to(device) if len(batch) > 3 else None
            prediction = model(bev.to(device), scalar.to(device), weather)
            loss = loss_fn(prediction, action.to(device))
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimiser.step()
        train_loss = loss_on(model, train_loader, loss_fn, device)
        val_loss = loss_on(model, val_loader, loss_fn, device)
        print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({
                "model_state": model.state_dict(), "in_channels": int(train.bev.shape[1]),
                "weather_dim": weather_dim, "bev_pool_size": model.bev_pool_size,
                "scalar_mean": train.mean.tolist(), "scalar_std": train.std.tolist(),
                "best_val_loss": best_loss,
                "train_episodes": [path.name for path in train_files],
                "validation_episodes": [path.name for path in val_files],
                "seed": args.seed, "epochs": args.epochs, "epochs_trained": epoch,
                "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "patience": args.patience, "train_brake_samples": train_brake_samples,
                "validation_brake_samples": val_brake_samples,
                "min_brake_samples": args.min_brake_samples,
                "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
                "brake_weight": args.brake_weight,
            }, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    print(f"saved best checkpoint: {args.checkpoint}; validation loss={best_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
