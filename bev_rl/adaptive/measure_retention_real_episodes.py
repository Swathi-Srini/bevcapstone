"""READ-ONLY measurement: how often the GitHub YOLO detector still finds a vehicle in fog and rain, in REAL driving episodes at a given traffic density.

Why this exists. The first retention table (measure_paired.py) used scenes of density 0.3 in which up to three vehicles were moved to fixed spots 8-14 m in front of a
stationary car, and it was then applied to driving tests at densities 0.05 and 0.15. Whether a per-vehicle detection rate measured that way transfers to another traffic
density, to other distances and to vehicles that occlude each other was never tested. This script measures it directly.

What it does. The ego car drives MetaDrive's stock rule-based driver (IDMPolicy) through ordinary episodes at the chosen density (nothing is teleported). Every
--every steps it captures the front-left and front-right camera frames TWICE, one simulator step apart (a 'burst' of 2). For every vehicle that is truly visible in the
front-left image (projects into it, 1-30 m ahead, not hidden behind a nearer vehicle - the teammates' own visibility test) it records whether a YOLO box contains the
vehicle's projected centre, on the clean frame and on the SAME frame with the weather filter applied (paired). Rows keep the true distance, the number of visible vehicles
(a proxy for crowding) and the time within the burst, so retention can be computed by distance, by density, and for repeated misses of the same vehicle in consecutive steps.
Nothing in the teammates' modules is modified; their functions are only called.

    python -m bev_rl.adaptive.measure_retention_real_episodes --density 0.15 --seeds 8000-8029 --out bev_rl/adaptive/results/retention_real_d0.15.json
Seeds 8000+ are a fresh block (never used for training, validation, evaluation or any earlier measurement).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

CONDS = [("none", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]


def capture(md, yolo, ctx, seed, step, burst, density, rows):
    """Detect every truly visible vehicle on the clean frame and on each weather condition (paired, same frame)."""
    from manual_drive_stereo_yolo_weather import FOCAL_LENGTH_PX, apply_synchronized_stereo_weather, frame_from_sensor
    from ground_truth_stereo_benchmark import is_occluded_by_nearer_vehicle, project_world_point, visible_surface_point
    from yolo.yolo_utils import run_yolo
    raw_l, raw_r = frame_from_sensor(md, "front_left_camera"), frame_from_sensor(md, "front_right_camera")
    camera = md.engine.get_sensor("front_left_camera").get_cam()
    ego = np.asarray(md.agent.position, float)
    cands = []
    for v in md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
        if v is md.agent:
            continue
        surface, _ = visible_surface_point(v, camera, md.engine.render)
        proj = project_world_point(camera, md.engine.render, surface)
        if proj is None or not 1.0 <= proj[2] <= 30.0:
            continue
        cands.append({"vehicle": v, "u": proj[0], "v": proj[1], "forward_m": proj[2]})
    cands = [c for c in cands if not is_occluded_by_nearer_vehicle(c, cands, FOCAL_LENGTH_PX)]
    if not cands:
        return 0
    det = {}
    for kind, lvl in CONDS:
        np.random.seed(20261113 + seed * 1000 + step * 2 + burst)
        left = raw_l if kind == "none" else apply_synchronized_stereo_weather(raw_l, raw_r, kind, lvl)[0]
        boxes = run_yolo(yolo, left)
        det[f"{kind}:{lvl}"] = [any(d["xmin"] <= c["u"] <= d["xmax"] and d["ymin"] <= c["v"] <= d["ymax"] for d in boxes) for c in cands]
    for i, c in enumerate(cands):
        v = c["vehicle"]
        rows.append({"density": density, "seed": seed, "step": step, "burst": burst, "vehicle_id": str(getattr(v, "id", i)), "truth_m": float(c["forward_m"]),
                     "speed_mps": float(v.speed), "n_visible": len(cands), "detected": {k: bool(val[i]) for k, val in det.items()}})
    return len(cands)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--density", type=float, required=True); p.add_argument("--seeds", default="8000-8029")
    p.add_argument("--every", type=int, default=25); p.add_argument("--max-steps", type=int, default=400); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    import torch, ultralytics  # noqa
    from bev_state import BEVStateAssembler          # importing it puts the YOLO+weather module on sys.path
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import mount_camera_rig
    from yolo.yolo_utils import ensure_yolo_model
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)
    md = make_env(use_idm=True, render=False, traffic_density=a.density, horizon=a.max_steps)
    rows, t0, zero = [], time.time(), np.array([0.0, 0.0], np.float32)
    try:
        for seed in range(lo, hi + 1):
            md.reset(seed=seed)
            mount_camera_rig(md)                      # every reset, as in the teammates' benchmark (see measure_depth_error.py)
            md.step(zero)
            done, step, n_in_scene = False, 1, 0
            while not done and step < a.max_steps:
                if step % a.every == 0:
                    for burst in (0, 1):
                        n_in_scene += capture(md, yolo, None, seed, step, burst, a.density, rows)
                        _, _, te, tr, _ = md.step(zero); step += 1
                        done = te or tr
                        if done:
                            break
                else:
                    _, _, te, tr, _ = md.step(zero); step += 1
                    done = te or tr
            print("scene", seed, f"steps={step} vehicle_samples={n_in_scene} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"density": a.density, "seeds": a.seeds, "every": a.every, "conds": CONDS, "rows": rows}))
    finally:
        md.close()


if __name__ == "__main__":
    main()
