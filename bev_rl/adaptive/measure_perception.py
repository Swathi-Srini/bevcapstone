"""READ-ONLY measurement: does the GitHub weather augmentation degrade the GitHub YOLO+stereo -> BEV pipeline?

Uses ONLY unmodified repository modules (tracked bev_policy.runtime.make_env, BEVStateAssembler, weather.weather_utils,
apply_synchronized_stereo_weather, run_yolo), following bev_policy/evaluate.py's weathered-perception path exactly.
One traffic vehicle is teleported to known ego-frame positions (ego stationary) so ground truth is exact; truth is used for
scoring only. Full camera resolution (no change).

    python -m bev_rl.adaptive.measure_perception --out bev_rl/adaptive/results/perception_degradation.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

PLACEMENTS = [(0.0, 6.0), (0.0, 10.0), (0.0, 14.0), (3.5, 10.0), (-3.5, 10.0)]
CONDITIONS = [("none", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1000); p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--weather-seeds", type=int, default=3); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch, ultralytics  # noqa  (DLL order)
    from bev_state import BEVStateAssembler
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import (CAMERA_RIGS, YOLO_CAMERAS, apply_synchronized_stereo_weather, frame_from_sensor)
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)   # same confidence as the repo evaluator
    asm = BEVStateAssembler()
    md = make_env(use_idm=False, render=False, traffic_density=a.density, horizon=200)
    rows, t0 = [], time.time()
    try:
        for (tx, ty) in PLACEMENTS:
            md.reset(seed=a.seed)
            _, info = md.reset(seed=a.seed)
            asm.mount_if_needed(md)
            ego, th = np.asarray(md.agent.position, float), float(md.agent.heading_theta)
            right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
            veh = min((o for o in md.engine.get_objects(lambda o: hasattr(o, "speed")).values() if o is not md.agent),
                      key=lambda o: np.hypot(*(np.asarray(o.position) - ego)))
            veh.set_position(ego + tx * right + ty * fwd); veh.set_heading_theta(th); veh.set_velocity(fwd, 0.0)
            _, _, _, _, info = md.step(np.array([0.0, 0.0], np.float32))
            ego, th = np.asarray(md.agent.position, float), float(md.agent.heading_theta)
            right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
            d = np.asarray(veh.position, float) - ego
            gt = (float(d @ right), float(d @ fwd))
            raw = {n: frame_from_sensor(md, n) for n in CAMERA_RIGS}
            clean_bev = None
            for kind, lvl in CONDITIONS:
                for ws in range(1 if kind == "none" else a.weather_seeds):
                    np.random.seed(20260829 + a.seed + ws)
                    if kind == "none":
                        frames = raw
                    else:
                        frames = {n: apply_weather(f, kind, lvl) for n, f in raw.items()}
                        frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(
                            raw["front_left_camera"], raw["front_right_camera"], kind, lvl)
                    dets = {n: run_yolo(yolo, frames[n]) for n in YOLO_CAMERAS}
                    st = asm.assemble(env=md, info=info, detections_by_camera=dets, frames=frames)
                    objs = [(float(o.x_right_m), float(o.y_forward_m)) for o in st.objects]
                    err = min((np.hypot(x - gt[0], y - gt[1]) for x, y in objs), default=None)
                    bev = np.asarray(st.bev_grid)
                    if kind == "none":
                        clean_bev = bev
                    occ = int((bev == 1.0).sum())
                    rows.append({"placed": [tx, ty], "gt": list(gt), "weather": kind, "level": lvl, "weather_seed": ws,
                                 "n_yolo": sum(len(v) for v in dets.values()), "n_objects": len(objs),
                                 "front_yolo": len(dets.get("front_left_camera", [])),
                                 "err_m": None if err is None else float(err), "detected_3m": bool(err is not None and err < 3.0),
                                 "bev_occupied_cells": occ, "bev_changed_cells_vs_clean": int((bev != clean_bev).sum())})
                    print(rows[-1], f"t={time.time()-t0:.0f}s", flush=True)
    finally:
        md.close()
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["weather"], r["level"])].append(r)
    summary = {f"{k[0]}:{k[1]}": {"n": len(v), "detect_rate_3m": float(np.mean([x["detected_3m"] for x in v])),
                                  "mean_err_m": float(np.mean([x["err_m"] for x in v if x["err_m"] is not None])) if any(x["err_m"] is not None for x in v) else None,
                                  "mean_bev_changed_cells": float(np.mean([x["bev_changed_cells_vs_clean"] for x in v]))} for k, v in g.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"seed": a.seed, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
