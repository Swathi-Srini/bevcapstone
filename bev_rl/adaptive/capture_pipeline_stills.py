"""STILL-FRAME CAPTURE of the live perception pipeline under different weather.   NEW FILE (2026-11-17).  Read-only: no model, estimator, seed or evaluation file is changed.

One scene (scenario seed, density 0.05 = sparse traffic) is driven with the unmodified Model A in clear weather up to step T.  At that step the five RAW camera frames are taken ONCE, then every weather
setting is applied to the SAME raw frames with the same function and the same noise seed the evaluation uses (np.random.seed(20261114 + global_seed * 100003 + T)).  For each setting we run the
genuine pipeline: synchronised stereo weather -> SGM depth -> YOLOv8n on 4 views -> object footprints -> BEV builder.  The privileged simulator BEV (truth objects) is saved as a reference only.
So the only thing that differs between columns of the figure is the weather.

    python -m bev_rl.adaptive.capture_pipeline_stills --seed 5219 --step 195 --out report/paper_severity/figures/capture
Output (in --out): cam_<name>.png (raw), <cond>_front_left.png, <cond>_depth.npy, <cond>_dets.json, <cond>_bev.npy, truth_bev.npy, stats.json
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bev_rl.adaptive import camera_dagger as cd

A_CKPT = "bev_rl/adaptive/results/camera_dagger_final/e2e_ov_split_final/best.pt"
CONDS = [("clear", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=5219); ap.add_argument("--step", type=int, default=195); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cd.verify_frozen()
    import torch
    torch.set_num_threads(1)
    from bev_rl.adaptive.env import SCALE
    import bev_state.assembler as A
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
    from weather.weather_utils import apply_weather
    net, mu, sd = cd.load_net(A_CKPT, False, torch)
    env = cd.make_env(0.05, 1600, 99, sensor=False)
    o, _ = env.reset(options={"scenario_seed": a.seed, "weather": ("clear", 0.0)}); env.drng = np.random.default_rng(a.seed * 7 + 1)
    while env.step_count < a.step:
        bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); act = cd.student_act(net, False, torch, bev, raw, 0.0, mu, sd)
        o, r, te, tr, info = env.step(act)
        if te or tr: raise SystemExit(f"episode ended at step {env.step_count} before the capture step")
    T = env.step_count
    raw_frames = {n: frame_from_sensor(env.md, n).copy() for n in CAMERA_RIGS}
    for n, f in raw_frames.items(): cv2.imwrite(str(a.out / f"cam_{n}.png"), f)
    truth = env.builder.build(env.truth_objects(), env.md); np.save(a.out / "truth_bev.npy", truth)
    stats = {"seed": a.seed, "step": T, "density": 0.05, "truth_objects_in_window": len(env.truth_objects()), "conditions": {}}
    for kind, level in CONDS:
        tag = f"{kind}_{level}"
        if kind == "clear": frames = {n: f.copy() for n, f in raw_frames.items()}
        else:
            np.random.seed(20261114 + int(env.md.engine.global_seed) * 100003 + T)
            frames = {n: apply_weather(f.copy(), kind, level) for n, f in raw_frames.items()}
            frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw_frames["front_left_camera"].copy(), raw_frames["front_right_camera"].copy(), kind, level)
        depth = A.front_stereo_depth(frames["front_left_camera"], frames["front_right_camera"], env.asm.matcher)
        dets = {n: A.run_yolo(env.yolo, frames[n]) for n in A.YOLO_CAMERAS}
        objs = env.asm.objects_from_detections(dets, depth); bev = env.builder.build(objs, env.md)
        cv2.imwrite(str(a.out / f"{tag}_front_left.png"), frames["front_left_camera"]); np.save(a.out / f"{tag}_depth.npy", depth); np.save(a.out / f"{tag}_bev.npy", bev)
        (a.out / f"{tag}_dets.json").write_text(json.dumps(dets))
        h = depth.shape[0]; road = depth[h // 2:, :]
        stats["conditions"][tag] = {"valid_depth_frac_lower_half": float((road > 0).mean()), "valid_depth_frac_all": float((depth > 0).mean()), "n_detections": {n: len(d) for n, d in dets.items()},
                                    "n_objects_in_bev": len(objs), "bev_obstacle_cells": int((bev >= 0.99).sum()), "bev_obstacle_cells_truth": int((truth >= 0.99).sum())}
        print(tag, json.dumps(stats["conditions"][tag]), flush=True)
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1)); env.md.close()


if __name__ == "__main__":
    main()
