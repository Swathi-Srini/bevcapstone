"""READ-ONLY paired degradation curve of the GitHub YOLO+stereo->BEV pipeline under the GitHub weather augmentation.

Each scene: up to 3 traffic vehicles teleported around the stationary ego; the SAME frames are perceived clean and with each
weather condition (paired). Reported: YOLO detections per frame and BEV objects, as a fraction of the clean count.
No module is modified; full camera resolution; truth is not used for scoring (retention only).

    python -m bev_rl.adaptive.measure_paired --out bev_rl/adaptive/results/perception_curve.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

SPOTS = [(0.0, 10.0), (3.5, 14.0), (-3.5, 8.0)]
CONDS = [("fog", l) for l in (0.15, 0.3, 0.45, 0.6, 0.8, 1.0)] + [("rain", l) for l in (0.1, 0.2, 0.3, 0.45, 0.6, 1.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1000-1019"); p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--weather-seeds", type=int, default=2); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    import torch, ultralytics  # noqa
    from bev_state import BEVStateAssembler
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, YOLO_CAMERAS, apply_synchronized_stereo_weather, frame_from_sensor
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)
    asm = BEVStateAssembler()
    md = make_env(use_idm=False, render=False, traffic_density=a.density, horizon=200)
    rows, t0 = [], time.time()
    try:
        for seed in range(lo, hi + 1):
            _, info = md.reset(seed=seed)
            asm.mount_if_needed(md)
            ego, th = np.asarray(md.agent.position, float), float(md.agent.heading_theta)
            right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
            vehs = sorted((o for o in md.engine.get_objects(lambda o: hasattr(o, "speed")).values() if o is not md.agent),
                          key=lambda o: np.hypot(*(np.asarray(o.position) - ego)))[:len(SPOTS)]
            for v, (tx, ty) in zip(vehs, SPOTS):
                v.set_position(ego + tx * right + ty * fwd); v.set_heading_theta(th); v.set_velocity(fwd, 0.0)
            _, _, _, _, info = md.step(np.array([0.0, 0.0], np.float32))
            raw = {n: frame_from_sensor(md, n) for n in CAMERA_RIGS}
            for kind, lvl in [("none", 0.0)] + CONDS:
                for ws in range(1 if kind == "none" else a.weather_seeds):
                    np.random.seed(20260829 + seed * 10 + ws)
                    if kind == "none":
                        frames = raw
                    else:
                        frames = {n: apply_weather(f, kind, lvl) for n, f in raw.items()}
                        frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(
                            raw["front_left_camera"], raw["front_right_camera"], kind, lvl)
                    dets = {n: run_yolo(yolo, frames[n]) for n in YOLO_CAMERAS}
                    st = asm.assemble(env=md, info=info, detections_by_camera=dets, frames=frames)
                    rows.append({"seed": seed, "weather": kind, "level": lvl, "ws": ws, "n_yolo": sum(len(v) for v in dets.values()),
                                 "front": len(dets["front_left_camera"]), "n_objects": len(st.objects)})
            print("scene", seed, f"t={time.time()-t0:.0f}s", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({"rows": rows}, indent=1))
    finally:
        md.close()


if __name__ == "__main__":
    main()
