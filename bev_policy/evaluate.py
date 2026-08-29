"""Evaluate a trained BEV behavioural-cloning checkpoint in MetaDrive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bev_state import BEVStateAssembler
from bev_policy.model import BEVScalarPolicy
from bev_policy.runtime import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("bev_policy/checkpoints/bc_clear.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--record-video", type=Path, default=None,
                        help="Optional MP4 of the front-left camera for smooth post-run playback.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    policy = BEVScalarPolicy(saved["in_channels"]).to(device)
    policy.load_state_dict(saved["model_state"]); policy.eval()
    mean, std = np.asarray(saved["scalar_mean"], dtype=np.float32), np.asarray(saved["scalar_std"], dtype=np.float32)
    from yolo.yolo_utils import ensure_yolo_model
    yolo = ensure_yolo_model(str(device), args.yolo_model, None, 0.4)
    env, assembler = make_env(use_idm=False, render=args.render, traffic_density=args.traffic_density, horizon=args.max_steps), BEVStateAssembler()
    results = []
    writer = None
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode; _, info = env.reset(seed=seed)
            speeds, final_info = [], dict(info)
            for step in range(args.max_steps):
                state = assembler.assemble(env=env, yolo_model=yolo, info=info)
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
                speeds.append(float(getattr(env.agent, "speed_km_h", 0.0)) / 3.6); final_info = dict(info)
                if args.render:
                    env.render(text={"mode": "BEV BC evaluation", "seed": seed, "step": step})
                if terminated or truncated:
                    break
            outcome = "success" if final_info.get("arrive_dest") else "collision" if (final_info.get("crash_vehicle") or final_info.get("crash_object")) else "off_road" if final_info.get("out_of_road") else "max_step"
            row = {"seed": seed, "outcome": outcome, "steps": len(speeds), "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0}
            results.append(row); print(row)
    finally:
        env.close()
        if writer is not None:
            writer.release()
    print(json.dumps({"episodes": len(results), "success_rate": float(np.mean([r["outcome"] == "success" for r in results])),
                      "collision_rate": float(np.mean([r["outcome"] == "collision" for r in results])),
                      "off_road_rate": float(np.mean([r["outcome"] == "off_road" for r in results])),
                      "mean_speed_mps": float(np.mean([r["mean_speed_mps"] for r in results]))}, indent=2))
    if args.record_video:
        print(f"Smooth playback video: {args.record_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
