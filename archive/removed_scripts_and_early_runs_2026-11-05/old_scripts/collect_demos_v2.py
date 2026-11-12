"""IDM expert demonstrations with CAMERA-DERIVED BEV v2 observations under a rotating weather schedule.

    python -m bev_policy.collect_demos_v2 --seed 4000 --episodes 40 --out bev_policy/data/v2

The expert (MetaDrive IDM) uses simulator state, as in the original BC set; the *observation* stored for the learner is
the camera BEV v2 (+ the 6-D scalar and the explicit weather vector). Expert actions are weather-independent. Episodes
cycle through the schedule below so weather types/intensities are balanced; the schedule index is stored.
Zero traffic -> YOLO is disabled (the obstacle channel is empty by construction in these scenes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEDULE = [("clear", 0.0), ("fog", 0.3), ("rain", 0.3), ("clear", 0.0), ("fog", 0.6), ("rain", 0.6), ("fog", 0.9), ("rain", 0.9)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=4000); p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=400); p.add_argument("--out", type=Path, default=Path("bev_policy/data/v2"))
    p.add_argument("--cam-scale", type=float, default=0.5)
    a = p.parse_args()
    import bev_state  # noqa: F401
    import torch, ultralytics  # noqa: F401,E401
    from bev_policy.runtime import make_env
    from bev_rl.perception_v2 import CameraBEVPerception
    from bev_state import BEVStateAssembler
    from bev_state.validate_bev import free_gb
    from bev_state.weather import weather_vector
    a.out.mkdir(parents=True, exist_ok=True)
    perc = CameraBEVPerception(None, BEVStateAssembler(), use_yolo=False, base_seed=a.seed)
    env = make_env(use_idm=True, render=False, traffic_density=0.0, horizon=a.max_steps, cam_scale=a.cam_scale)
    try:
        for i in range(a.episodes):
            if free_gb() < 1.0:
                print("STOP: low memory", flush=True); break
            seed, (kind, lvl) = a.seed + i, SCHEDULE[i % len(SCHEDULE)]
            out = a.out / f"seed_{seed:05d}.npz"
            if out.exists():
                continue
            _, info = env.reset(seed=seed); perc.on_reset(env); perc.set_weather(kind, lvl)
            bevs, sc, acts = [], [], []
            for _ in range(a.max_steps):
                bev, scalar = perc(env, info)
                _, _, term, trunc, info = env.step([0.0, 0.0])
                act = np.clip(np.asarray(env.agent.current_action, np.float32), -1, 1)
                bevs.append(bev.astype(np.uint8)); sc.append(scalar); acts.append(act)
                if term or trunc:
                    break
            n = len(acts)
            np.savez_compressed(out, bev=np.stack(bevs), scalar=np.stack(sc).astype(np.float32), action=np.stack(acts),
                                weather=np.tile(weather_vector(kind, lvl), (n, 1)),
                                metadata_json=np.asarray(json.dumps({"seed": seed, "weather": [kind, lvl], "expert": "MetaDrive_IDMPolicy",
                                    "cam_scale": a.cam_scale, "arrive_dest": bool(info.get("arrive_dest", False)),
                                    "termination": {k: bool(info.get(k, False)) for k in ("arrive_dest", "crash_vehicle", "out_of_road", "max_step")}})))
            print(f"saved {out.name} {kind}:{lvl} samples={n} arrive={info.get('arrive_dest')}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
