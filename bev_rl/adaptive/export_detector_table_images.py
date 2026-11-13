"""Regenerate and SAVE the camera pictures behind the detector retention table (retention_table.json / perception_curve.json).

measure_paired.py counted detections but never saved the pictures. This script repeats exactly the same procedure (same scenes = seeds 1000-1019 at traffic density 0.3,
same three vehicle spots in front of the stationary ego car, same weather noise seed 20260829 + seed*10 + weather draw, same detector) and writes, for every scene,
every weather condition and every YOLO camera (front-left, left, right, rear), a JPEG picture with the detector's boxes drawn on it, plus a counts table.
It then compares its detection counts with the ones stored in perception_curve.json, so the folder can be trusted to be the same data.

    python -m bev_rl.adaptive.export_detector_table_images --out report/caution_model_detector_table_images
Nothing in the teammates' modules is modified; their functions are only called.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

SPOTS = [(0.0, 10.0), (3.5, 14.0), (-3.5, 8.0)]
CONDS = [("fog", l) for l in (0.15, 0.3, 0.45, 0.6, 0.8, 1.0)] + [("rain", l) for l in (0.1, 0.2, 0.3, 0.45, 0.6, 1.0)]


def draw(img, dets, title):
    out = img.copy()
    for d in dets:
        x0, y0, x1, y1 = (int(d[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 220, 0), 3)
        cv2.putText(out, f"{d.get('label', 'vehicle')} {float(d.get('confidence', 0)):.2f}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, f"{title}   detections: {len(dets)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="1000-1019"); p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--weather-seeds", type=int, default=2); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=Path("bev_rl/adaptive/results/perception_curve.json"))
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    import torch, ultralytics  # noqa
    from bev_state import BEVStateAssembler          # importing it puts the YOLO+weather module on sys.path
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, YOLO_CAMERAS, apply_synchronized_stereo_weather, frame_from_sensor
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4)
    md = make_env(use_idm=False, render=False, traffic_density=a.density, horizon=200)
    asm = BEVStateAssembler()
    a.out.mkdir(parents=True, exist_ok=True)
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
            md.step(np.array([0.0, 0.0], np.float32))
            raw = {n: frame_from_sensor(md, n) for n in CAMERA_RIGS}
            sdir = a.out / f"scene_seed_{seed}"; sdir.mkdir(exist_ok=True)
            for kind, lvl in [("none", 0.0)] + CONDS:
                for ws in range(1 if kind == "none" else a.weather_seeds):
                    np.random.seed(20260829 + seed * 10 + ws)
                    if kind == "none":
                        frames = raw
                    else:
                        frames = {n: apply_weather(f, kind, lvl) for n, f in raw.items()}
                        frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(
                            raw["front_left_camera"], raw["front_right_camera"], kind, lvl)
                    tag = "clear" if kind == "none" else f"{kind}_{lvl}__draw{ws}"
                    per_cam = {}
                    for n in YOLO_CAMERAS:
                        dets = run_yolo(yolo, frames[n]); per_cam[n] = len(dets)
                        cv2.imwrite(str(sdir / f"{tag}__{n.replace('_camera', '')}.jpg"), draw(frames[n], dets, f"seed {seed}  {tag}  {n}"), [cv2.IMWRITE_JPEG_QUALITY, 88])
                    rows.append({"seed": seed, "weather": kind, "level": lvl, "ws": ws, "n_yolo": sum(per_cam.values()), "front": per_cam["front_left_camera"],
                                 **{f"n_{n.replace('_camera', '')}": c for n, c in per_cam.items()}})
            print("scene", seed, f"t={time.time() - t0:.0f}s", flush=True)
    finally:
        md.close()
    with open(a.out / "detection_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ref = {(r["seed"], r["weather"], r["level"], r["ws"]): r for r in json.loads(a.reference.read_text())["rows"]}
    same = sum(1 for r in rows if (r["seed"], r["weather"], r["level"], r["ws"]) in ref and ref[(r["seed"], r["weather"], r["level"], r["ws"])]["n_yolo"] == r["n_yolo"]
               and ref[(r["seed"], r["weather"], r["level"], r["ws"])]["front"] == r["front"])
    msg = f"{same} of {len(rows)} (scene, weather, draw) cells give the same detection counts as perception_curve.json"
    print(msg); (a.out / "reproduction_check.txt").write_text(msg + "\n")


if __name__ == "__main__":
    main()
