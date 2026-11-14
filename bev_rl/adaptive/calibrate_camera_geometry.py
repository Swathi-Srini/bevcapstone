"""CONTROLLED PLACEMENT experiment: what does the unmodified camera pipeline report for a vehicle at a KNOWN position around the ego?

For each scene seed, K vehicles are placed one at a time at every point of a grid (lateral x in [-8, 8] m, forward y in [-2, 16] m, lane-aligned, stationary), all other vehicles are moved
far away. The full unmodified chain runs on the five rendered pictures (clear weather: geometry is calibrated without weather): YOLOv8n on four views + front stereo depth +
BEVStateAssembler.objects_from_detections. Stored per placement: true centre (x, y), true LENGTH / WIDTH, and every object the pipeline returned (camera, depth method, label, confidence,
reported x, y). Ground truth is used HERE ONLY, offline, to fit the builder's geometry corrections; the builder itself never sees it.

    python -m bev_rl.adaptive.calibrate_camera_geometry --seeds 2000 2001 --k 2 --out bev_rl/adaptive/results/camera_geometry
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
XS = [-8.0, -5.5, -3.5, 0.0, 3.5, 5.5, 8.0]
YS = [-2.0, 1.0, 4.0, 7.0, 10.0, 13.0, 16.0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--k", type=int, default=2); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    import torch, ultralytics  # noqa
    from bev_state import BEVStateAssembler
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, frame_from_sensor
    from yolo.yolo_utils import ensure_yolo_model
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)
    md = make_env(use_idm=False, render=False, traffic_density=0.3, horizon=200); asm = BEVStateAssembler()
    for seed in a.seeds:
        out = a.out / f"placements_seed{seed}.jsonl"
        if out.exists(): continue
        t0 = time.time(); md.reset(seed=seed); asm.mount_if_needed(md)
        ego = np.asarray(md.agent.position, float); th = float(md.agent.heading_theta)
        right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
        vehs = sorted((o for o in md.engine.get_objects(lambda o: hasattr(o, "speed")).values() if o is not md.agent), key=lambda o: np.hypot(*(np.asarray(o.position) - ego)))
        rows = []
        for vi, mover in enumerate(vehs[:a.k]):
            for ty in YS:
                for tx in XS:
                    for i, v in enumerate(vehs):
                        v.set_velocity(fwd, 0.0)
                        if v is mover: v.set_position(ego + tx * right + ty * fwd); v.set_heading_theta(th)
                        else: v.set_position(ego + (300 + 12 * i) * fwd + 40 * right)
                    zero = np.array([0.0, 0.0], np.float32); md.step(zero); md.step(zero)
                    ego_now, th_now = np.asarray(md.agent.position, float), float(md.agent.heading_theta)
                    r_now, f_now = np.array([np.sin(th_now), -np.cos(th_now)]), np.array([np.cos(th_now), np.sin(th_now)])
                    d = np.asarray(mover.position, float) - ego_now
                    raw = {n: frame_from_sensor(md, n) for n in CAMERA_RIGS}
                    obs = asm.assemble(env=None, yolo_model=yolo, frames=raw)
                    rows.append({"seed": seed, "vehicle": vi, "true_x": float(d @ r_now), "true_y": float(d @ f_now), "length": float(getattr(mover, "LENGTH", 4.5)), "width": float(getattr(mover, "WIDTH", 1.85)),
                                 "objects": [{"camera": o.camera, "method": o.depth_method, "label": o.label, "conf": o.confidence, "x": o.x_right_m, "y": o.y_forward_m} for o in obs.objects]})
            print("seed", seed, "vehicle", vi, "done, placements so far", len(rows), f"{time.time() - t0:.0f}s", flush=True)
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    md.close()


if __name__ == "__main__":
    main()
