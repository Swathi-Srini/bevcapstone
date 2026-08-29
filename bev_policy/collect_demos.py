"""Automatically collect clear-condition BEV demonstrations from IDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bev_state import BEVStateAssembler
from bev_policy.runtime import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("bev_policy/data/clear"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or not (10 <= args.seed and args.seed + args.episodes <= 10010):
        raise ValueError("Use positive episode/step counts and MetaDrive seeds in [10, 10010).")

    from yolo.yolo_utils import ensure_yolo_model
    model = ensure_yolo_model(args.device, args.yolo_model, None, 0.4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env, assembler = make_env(use_idm=True, render=args.render, traffic_density=args.traffic_density, horizon=args.max_steps), BEVStateAssembler()
    print("IDM is collecting clear-condition demonstration labels. The ego vehicle drives automatically.")
    try:
        for index in range(args.episodes):
            seed = args.seed + index
            _, info = env.reset(seed=seed)
            bevs, states, actions, final_info = [], [], [], dict(info)
            for _ in range(args.max_steps):
                observation = assembler.assemble(env=env, yolo_model=model, info=info)
                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                action = np.clip(np.asarray(env.agent.current_action, dtype=np.float32), -1.0, 1.0)
                bevs.append(observation.bev_grid); states.append(observation.scalar_state); actions.append(action)
                final_info = dict(info)
                if args.render:
                    env.render(text={"mode": "IDM BEV demo collection", "seed": seed, "samples": len(actions)})
                if terminated or truncated:
                    break
            output = args.output_dir / f"seed_{seed:05d}.npz"
            np.savez_compressed(output, bev=np.stack(bevs), scalar=np.stack(states), action=np.stack(actions),
                                metadata_json=np.asarray(json.dumps({"seed": seed, "expert": "MetaDrive_IDMPolicy",
                                    "traffic_density": args.traffic_density, "termination": {key: bool(final_info.get(key, False)) for key in ("arrive_dest", "crash_vehicle", "out_of_road", "max_step")} })))
            print(f"saved {output} ({len(actions)} aligned samples)")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
