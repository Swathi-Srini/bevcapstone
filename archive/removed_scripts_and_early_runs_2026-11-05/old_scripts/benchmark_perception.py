"""Per-stage latency / memory of the camera-BEV perception (one simulator, sequential conditions).

    python -m bev_rl.benchmark_perception --steps 30 --out bev_rl/results/perception_benchmark.json
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np


def rss_gb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=30); p.add_argument("--cam-scale", type=float, default=0.5); p.add_argument("--yolo", action="store_true"); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from bev_rl.factory import load_yolo
    from bev_rl.perception_v2 import CameraBEVPerception
    from bev_policy.runtime import make_env
    from bev_state import BEVStateAssembler
    asm = BEVStateAssembler()
    perc = CameraBEVPerception(load_yolo() if a.yolo else None, asm, use_yolo=a.yolo)
    md = make_env(use_idm=False, render=False, traffic_density=0.0, horizon=a.steps + 10, cam_scale=a.cam_scale)
    res = {}
    try:
        for kind, lvl in (("clear", 0.0), ("fog", 0.6), ("rain", 0.6)):
            md.reset(seed=1000); perc.on_reset(md); perc.set_weather(kind, lvl); info = {}
            perc.timing = {k: 0.0 for k in perc.timing}
            t0 = time.perf_counter(); sim = 0.0
            for _ in range(a.steps):
                s0 = time.perf_counter(); _, _, term, trunc, info = md.step(np.array([0.0, 0.3], np.float32)); sim += time.perf_counter() - s0
                bev, sc = perc(md, info)
                if term or trunc: break
            n = max(1, perc.timing["n"])
            res[f"{kind}:{lvl}"] = {"steps": int(n), "sim_step_s": sim / n, "total_per_step_s": (time.perf_counter() - t0) / n,
                                    **{k: perc.timing[k] / n for k in ("frames", "weather", "yolo", "stereo", "surface")}, "rss_gb": rss_gb()}
            print(kind, res[f"{kind}:{lvl}"], flush=True)
    finally:
        md.close()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
