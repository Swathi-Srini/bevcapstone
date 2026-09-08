"""Evaluate a trained BEV behavioural-cloning checkpoint in MetaDrive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bev_state import BEVStateAssembler
from bev_policy.model import BEVScalarPolicy
from bev_policy.metrics import action_metrics, summarise_episode_rows
from bev_policy.runtime import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("bev_policy/checkpoints/bc_clear.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--weather", choices=("none", "fog", "rain"), default="none",
                        help="Synthetic visual condition applied to the perception input.")
    parser.add_argument("--weather-level", type=float, default=0.5,
                        help="Synthetic fog/rain intensity in [0,1].")
    parser.add_argument("--visibility-range-m", type=float, default=None,
                        help="Privileged forward BEV visibility horizon in metres; not camera-inferred fog range.")
    parser.add_argument("--weather-seed", type=int, default=20260829,
                        help="Base seed for reproducible synthetic weather realizations.")
    parser.add_argument("--output-dir", type=Path, default=Path("bev_policy/outputs/weather"),
                        help="Directory for per-condition JSON and CSV results.")
    parser.add_argument("--record-video", type=Path, default=None,
                        help="Optional MP4 of the front-left camera for smooth post-run playback.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    if not 0.0 <= args.weather_level <= 1.0:
        raise ValueError("--weather-level must be between 0 and 1.")
    if args.visibility_range_m is not None and args.visibility_range_m < 0.0:
        raise ValueError("--visibility-range-m must be greater than or equal to zero.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = BEVScalarPolicy(saved["in_channels"]).to(device)
    policy.load_state_dict(saved["model_state"]); policy.eval()
    mean, std = np.asarray(saved["scalar_mean"], dtype=np.float32), np.asarray(saved["scalar_std"], dtype=np.float32)
    from manual_drive_stereo_yolo_weather import (
        CAMERA_RIGS, YOLO_CAMERAS, apply_synchronized_stereo_weather, frame_from_sensor,
    )
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    yolo = ensure_yolo_model(str(device), args.yolo_model, None, 0.4)
    env, assembler = make_env(use_idm=False, render=args.render, traffic_density=args.traffic_density, horizon=args.max_steps), BEVStateAssembler()
    results = []
    writer = None
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode; _, info = env.reset(seed=seed)
            # This affects only the synthetic image corruption. Route generation
            # remains controlled by the identical MetaDrive seed above.
            np.random.seed(args.weather_seed + seed)
            assembler.mount_if_needed(env)
            speeds, actions, final_info = [], [], dict(info)
            for step in range(args.max_steps):
                raw_frames = {name: frame_from_sensor(env, name) for name in CAMERA_RIGS}
                if args.weather == "none":
                    perception_frames = raw_frames
                else:
                    perception_frames = {name: apply_weather(frame, args.weather, args.weather_level)
                                         for name, frame in raw_frames.items()}
                    perception_frames["front_left_camera"], perception_frames["front_right_camera"] = (
                        apply_synchronized_stereo_weather(
                            raw_frames["front_left_camera"], raw_frames["front_right_camera"],
                            args.weather, args.weather_level
                        )
                    )
                detections = {name: run_yolo(yolo, perception_frames[name]) for name in YOLO_CAMERAS}
                state = assembler.assemble(
                    env=env, info=info, detections_by_camera=detections, frames=perception_frames,
                visibility_range_m=args.visibility_range_m,
                )
                if args.record_video:
                    import cv2
                    frame = state.frames["front_left_camera"]
                    if writer is None:
                        args.record_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(args.record_video), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video output: {args.record_video}")
                    writer.write(frame)
                bev = torch.from_numpy(state.bev_grid[None, None]).to(device)
                scalar = torch.from_numpy(((state.scalar_state - mean) / std)[None]).to(device)
                with torch.no_grad():
                    action = policy(bev, scalar).squeeze(0).cpu().numpy()
                _, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                actions.append(np.asarray(action, dtype=np.float32))
                speeds.append(float(getattr(env.agent, "speed_km_h", 0.0)) / 3.6); final_info = dict(info)
                if args.render:
                    env.render(text={"mode": "BEV BC evaluation", "seed": seed, "step": step})
                if terminated or truncated:
                    break
            outcome = "success" if final_info.get("arrive_dest") else "collision" if (final_info.get("crash_vehicle") or final_info.get("crash_object")) else "off_road" if final_info.get("out_of_road") else "max_step"
            row = {"seed": seed, "outcome": outcome, "steps": len(speeds),
                   "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                   **action_metrics(actions)}
            results.append(row); print(row)
    finally:
        env.close()
        if writer is not None:
            writer.release()
    summary = summarise_episode_rows(results)
    report = {"config": {"checkpoint": str(args.checkpoint), "episodes": args.episodes, "seed": args.seed,
                         "max_steps": args.max_steps, "traffic_density": args.traffic_density,
                         "weather": args.weather, "weather_level": args.weather_level,
                         "visibility_range_m": args.visibility_range_m,
                         "weather_seed": args.weather_seed,
                         "weather_definition": "Synthetic visual corruption is applied to all perception frames; it does not change vehicle dynamics or road friction.",
                         "visibility_definition": "When set, visibility_range_m is a privileged controlled-visibility intervention that masks forward BEV cells; it is not camera-inferred physical fog range."},
              "episodes": results, "summary": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visibility_stem = "default" if args.visibility_range_m is None else f"{args.visibility_range_m:.2f}m"
    traffic_stem = f"{args.traffic_density:.2f}"
    stem = f"{args.weather}_level_{args.weather_level:.2f}_traffic_{traffic_stem}_visibility_{visibility_stem}_seed_{args.seed}_n_{args.episodes}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ["seed", "outcome"])
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary, indent=2))
    print(f"Saved structured results: {json_path} and {csv_path}")
    if args.record_video:
        print(f"Smooth playback video: {args.record_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
