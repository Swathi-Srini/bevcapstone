"""Prove LazyStereoPerception == FullPerception (bit-identical BEV and scalars).

Uses a TRAFFIC scene driven by the IDM expert so the front camera really produces
detections (otherwise the stereo branch would never be exercised).
"""
import json
import sys
import time

import numpy as np

import torch  # noqa: F401  (import before MetaDrive)
from bev_rl.factory import FullPerception, LazyStereoPerception, load_yolo


def main(steps=120, seeds=(300, 301), density=0.3):
    from bev_policy.runtime import make_env
    from bev_state import BEVStateAssembler
    yolo = load_yolo()
    full, lazy = FullPerception(yolo, BEVStateAssembler()), LazyStereoPerception(yolo, BEVStateAssembler())
    env = make_env(use_idm=True, render=False, traffic_density=density, horizon=steps + 5)
    n = front = side_rear = identical = occupied_frames = 0
    t_full = t_lazy = 0.0
    try:
        for seed in seeds:
            _, info = env.reset(seed=seed)
            full.on_reset(env); lazy.on_reset(env)
            for _ in range(steps):
                t = time.time(); bev_f, sc_f = full(env, info); t_full += time.time() - t
                t = time.time(); bev_l, sc_l = lazy(env, info); t_lazy += time.time() - t
                n += 1
                same = np.array_equal(bev_f, bev_l) and np.array_equal(sc_f, sc_l)
                identical += int(same)
                occupied_frames += int((bev_f == 1.0).any())
                if not same:
                    print("MISMATCH at seed", seed, "frame", n, file=sys.stderr)
                _, _, term, trunc, info = env.step([0.0, 0.0])
                if term or trunc:
                    break
    finally:
        env.close()
    result = {"frames": n, "identical": identical, "frames_with_occupied_cells": occupied_frames,
              "mean_s_full": t_full / n, "mean_s_lazy": t_lazy / n}
    print(json.dumps(result, indent=2))
    return 0 if identical == n and occupied_frames > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
