"""READ-ONLY measurement: how the GitHub front-stereo depth + YOLO detection degrade under the GitHub weather augmentation, against MetaDrive truth.

Nothing in the teammates' modules is modified: their functions are only CALLED (apply_synchronized_stereo_weather, front_stereo_depth, object_stereo_depth,
run_yolo, project_world_point, visible_surface_point, sample_depth, is_occluded_by_nearer_vehicle).

Per scene: up to --n-veh traffic vehicles are teleported in front of the stationary ego at random forward distance 5-19 m and lateral offset {-3.5, 0, +3.5} (+-0.5 m).
For every vehicle that is visible (projects into the front-left image, 1-30 m, not occluded) and for every condition (clean, fog 0.6/0.8/1.0, rain 0.3/0.6/1.0) x weather draws:
    truth_m   : camera-forward distance to the vehicle's visible rear surface (MetaDrive pose; the teammates' reference definition)
    sgbm_m    : stereo depth sampled at the projected truth pixel (teammates' sample_depth)   -> depth validity / error independent of YOLO
    detected  : a YOLO box contains the projected truth pixel
    pipe_m    : the pipeline's own estimate for that box (object_stereo_depth: median valid stereo depth inside the box); None => 'stereo_sgm_invalid'
    python -m bev_rl.adaptive.measure_depth_error --seeds 7000-7039 --out bev_rl/adaptive/results/depth_error_measurements.json
Seeds 7000+ are a fresh block (never used for training/validation/evaluation). Camera-in-the-loop: ~1-5 s per condition, ~4 GB RAM.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

CONDS = [("none", 0.0)] + [("fog", l) for l in (0.6, 0.8, 1.0)] + [("rain", l) for l in (0.3, 0.6, 1.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="7000-7039"); p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--weather-seeds", type=int, default=2); p.add_argument("--n-veh", type=int, default=3); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    import torch, ultralytics  # noqa
    from bev_state import BEVStateAssembler                       # importing it puts the YOLO+weather module on sys.path (assembler.py, PERCEPTION_ROOT)
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import (CAMERA_RIGS, FOCAL_LENGTH_PX, apply_synchronized_stereo_weather, build_stereo_matcher,
                                                  frame_from_sensor, front_stereo_depth, mount_camera_rig, object_stereo_depth)
    from ground_truth_stereo_benchmark import is_occluded_by_nearer_vehicle, project_world_point, sample_depth, visible_surface_point
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)
    asm = BEVStateAssembler()
    md = make_env(use_idm=False, render=False, traffic_density=a.density, horizon=200)
    matcher = build_stereo_matcher()
    rng = np.random.default_rng(20261110)
    rows, t0 = [], time.time()
    try:
        for seed in range(lo, hi + 1):
            md.reset(seed=seed)
            mount_camera_rig(md)            # EVERY reset (as in the teammates' ground_truth_stereo_benchmark.py). asm.mount_if_needed mounts once per env object, so scenes 2..N had STALE cameras
                                            # (first measurement 2026-11-10 20:31: stereo valid at 2/73 clean samples, all in scene 7000) -> that run is invalid.
            ego, th = np.asarray(md.agent.position, float), float(md.agent.heading_theta)
            right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
            vehs = sorted((o for o in md.engine.get_objects(lambda o: hasattr(o, "speed")).values() if o is not md.agent),
                          key=lambda o: np.hypot(*(np.asarray(o.position) - ego)))[:a.n_veh]
            for v in vehs:
                y = float(rng.uniform(5.0, 19.0)); x = float(rng.choice([-3.5, 0.0, 3.5]) + rng.uniform(-0.5, 0.5))
                v.set_position(ego + x * right + y * fwd); v.set_heading_theta(th); v.set_velocity(fwd, 0.0)
            md.step(np.array([0.0, 0.0], np.float32))
            raw_l, raw_r = frame_from_sensor(md, "front_left_camera"), frame_from_sensor(md, "front_right_camera")
            camera = md.engine.get_sensor("front_left_camera").get_cam()
            cands = []
            for v in vehs:
                surface, half_len = visible_surface_point(v, camera, md.engine.render)
                proj = project_world_point(camera, md.engine.render, surface)
                if proj is None or not 1.0 <= proj[2] <= 30.0:
                    continue
                cands.append({"vehicle": v, "u": proj[0], "v": proj[1], "forward_m": proj[2]})
            cands = [c for c in cands if not is_occluded_by_nearer_vehicle(c, cands, FOCAL_LENGTH_PX)]
            for kind, lvl in CONDS:
                for ws in range(1 if kind == "none" else a.weather_seeds):
                    np.random.seed(20261110 + seed * 10 + ws)
                    L, R = (raw_l, raw_r) if kind == "none" else apply_synchronized_stereo_weather(raw_l, raw_r, kind, lvl)
                    depth = front_stereo_depth(L, R, matcher)
                    dets = run_yolo(yolo, L)
                    for c in cands:
                        det = next((d for d in dets if d["xmin"] <= c["u"] <= d["xmax"] and d["ymin"] <= c["v"] <= d["ymax"]), None)
                        sg = sample_depth(depth, c["u"], c["v"])
                        pm = object_stereo_depth(det, depth) if det is not None else None
                        rows.append({"seed": seed, "weather": kind, "level": lvl, "ws": ws, "truth_m": c["forward_m"], "u": c["u"], "v": c["v"],
                                     "sgbm_m": sg, "detected": det is not None, "pipe_m": pm, "n_dets": len(dets)})
            print("scene", seed, f"vehicles={len(cands)} rows={len(rows)} t={time.time() - t0:.0f}s", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"seeds": a.seeds, "density": a.density, "conds": CONDS, "rows": rows}))
    finally:
        md.close()


if __name__ == "__main__":
    main()
