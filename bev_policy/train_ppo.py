"""Fine-tune a BEV BC checkpoint with PPO under randomized camera weather."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bev_policy.ppo import BEVDrivingEnv, build_bc_initialized_ppo
from yolo.yolo_utils import ensure_yolo_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", type=Path,
                        default=Path("bev_policy/checkpoints/bc_sparse_traffic_5eps.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("bev_policy/checkpoints/ppo_bev_adaptive"))
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--weather", choices=("clear", "fog", "rain"), default="clear")
    parser.add_argument("--weather-level", type=float, default=0.8,
                        help="Fixed level, or maximum randomized level.")
    parser.add_argument("--random-weather", action=argparse.BooleanOptionalAction, default=True,
                        help="Randomize clear/fog/rain and intensity each episode (default: enabled).")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                        help="YOLO confidence threshold used while collecting PPO rollouts.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--target-kl", type=float, default=0.005,
                        help="Stop PPO epochs early when updates move too far from the rollout policy.")
    parser.add_argument("--bc-loss-coef", type=float, default=5.0,
                        help="Weight of the action-matching loss to the frozen BC policy.")
    parser.add_argument("--adaptive-speed-penalty", type=float, default=0.05,
                        help="Reward penalty per squared m/s above the BEV obstacle/weather speed envelope.")
    parser.add_argument("--collision-penalty", type=float, default=30.0,
                        help="Additional terminal penalty for collision or leaving the road.")
    parser.add_argument("--energy-reward-weight", type=float, default=0.0,
                        help="Reward penalty per estimated Wh of positive tractive work per step.")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.timesteps < 1 or args.max_steps < 1 or args.n_steps < 2 or args.inference_size < 32:
        raise ValueError("timesteps/max-steps must be positive, n-steps at least 2, and inference size at least 32.")
    if (
        args.batch_size < 1 or args.n_epochs < 1 or args.target_kl <= 0.0
        or args.learning_rate <= 0.0 or args.bc_loss_coef <= 0.0
        or args.adaptive_speed_penalty < 0.0 or args.collision_penalty <= 0.0
        or args.energy_reward_weight < 0.0
        or not 0.0 <= args.weather_level <= 1.0 or args.traffic_density < 0
        or not 0.0 <= args.confidence_threshold <= 1.0
    ):
        raise ValueError(
            "batch size/epochs/target KL/learning rate/BC loss coefficient must be positive, weather level in [0,1], "
            "confidence/weather in [0,1], penalties valid, and traffic non-negative."
        )
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {args.bc_checkpoint}")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    saved_bc = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=True)
    mean = np.asarray(saved_bc["scalar_mean"], dtype=np.float32)
    std = np.asarray(saved_bc["scalar_std"], dtype=np.float32)
    yolo = ensure_yolo_model(device, args.yolo_model, None, args.confidence_threshold)
    yolo.inference_size = args.inference_size
    env = BEVDrivingEnv(
        yolo, mean, std, weather=args.weather, weather_level=args.weather_level,
        random_weather=args.random_weather, traffic_density=args.traffic_density,
        max_steps=args.max_steps, render=args.render, seed=args.seed,
        adaptive_speed_penalty=args.adaptive_speed_penalty,
        collision_penalty=args.collision_penalty,
        energy_reward_weight=args.energy_reward_weight,
    )
    try:
        model = build_bc_initialized_ppo(
            env, args.bc_checkpoint, device=device, seed=args.seed,
            n_steps=args.n_steps, batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_epochs=args.n_epochs, target_kl=args.target_kl,
            bc_loss_coef=args.bc_loss_coef,
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output))
    finally:
        env.close()

    zip_path = args.output if args.output.suffix == ".zip" else Path(str(args.output) + ".zip")
    metadata = {
        "algorithm": "PPO",
        "initial_bc_checkpoint": str(args.bc_checkpoint),
        "scalar_mean": mean.tolist(),
        "scalar_std": std.tolist(),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "n_epochs": args.n_epochs,
        "target_kl": args.target_kl,
        "bc_loss_coef": args.bc_loss_coef,
        "adaptive_speed_penalty": args.adaptive_speed_penalty,
        "collision_penalty": args.collision_penalty,
        "energy_reward_weight": args.energy_reward_weight,
        "adaptive_reward": "BEV-only nearest front obstacle distance plus synthetic visibility horizon define a target speed; penalize overspeed squared. No NPC positions are used.",
        "energy_reward": "Physics-inspired positive tractive work per step from speed/acceleration; flat road, 1800 kg, Cd 0.23, A 2.22 m^2, rho 1.225 kg/m^3, Crr 0.01, dt 0.1s, no regeneration or grade. This is a simulator proxy, not real battery energy.",
        "bc_anchor": "Frozen BC policy action-matching MSE added to PPO actor loss.",
        "traffic_density": args.traffic_density,
        "confidence_threshold": args.confidence_threshold,
        "network": {
            "bev_cnn": "Conv(1,32,5x5) -> Conv(32,64,3x3) -> Conv(64,96,3x3) -> pool -> FC(96,128)",
            "scalar_mlp": "FC(6,64) -> FC(64,32)",
            "fusion": "128 BEV + 32 scalar = 160",
            "actor": "FC(160,128) -> FC(128,2) -> tanh; PPO gSDE exploration",
            "critic": "FC(160,128) -> FC(128,1)",
        },
        "action": "[steering, longitudinal throttle/brake], each in [-1,1]",
        "weather": args.weather,
        "weather_level_max": args.weather_level,
        "random_weather": args.random_weather,
        "weather_model": "Depth-dependent camera-weather approximation; vehicle dynamics and road friction unchanged.",
    }
    zip_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved PPO policy: {zip_path}")
    print(f"Saved normalization and training metadata: {zip_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
