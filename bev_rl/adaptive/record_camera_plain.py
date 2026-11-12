"""PLAIN front-camera videos in the format of bev_policy/outputs/videos/fog_seed1000.mp4 (teammates' reference):
raw 1200x900 front-left camera frame, GitHub weather augmentation applied (apply_weather, fresh random draw per frame as in their pipeline), written at 10 fps = real time,
ONE frame per simulation step, NO text / boxes / BEV. Each model drives the same scenario; file name carries model, seed and weather.

    python -m bev_rl.adaptive.record_camera_plain --seeds 1039 --weather fog:0.5 --arms ov_split repo_bc_sparse --out paper/videos_front_camera

The policies do NOT see these pixels: they drive from the simulator-derived BEV (weather = detector dropout from the measured retention table, no weather physics); the video shows the
front camera under the same weather level. Scenario = the evaluation environment (same MetaDrive config, density 0.05, horizon 800) + one front camera, re-mounted after every reset.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2
import numpy as np

ARMS = ["ov_split", "repo_bc_sparse", "repo_bc_clear", "base_a", "ctrl_nocaut", "caut_c0", "caut_auto", "safe_student", "stock_idm", "safe_teacher",
        "arch_bev_only", "arch_scalar_only", "ov_both", "ov_bev_only"]


def record(arm, seed, weather, out, env, torch, max_steps=None):
    from bev_rl.adaptive.caution import c_auto, keep_prob, prep
    from bev_rl.adaptive.degraded_env import make_safe_expert
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.record_videos import build
    from bev_rl.adaptive.spec_bev import overlay_github_bev
    from manual_drive_stereo_yolo_weather import frame_from_sensor
    from weather.weather_utils import apply_weather
    P = build(arm, torch, env); kind = P["kind"]
    o, _ = env.reset(options={"scenario_seed": seed, "weather": weather}); env.drng = np.random.default_rng(seed * 7 + 1)
    pol = None
    if kind == "stock_idm":
        from metadrive.policy.idm_policy import IDMPolicy
        pol = IDMPolicy(env.md.agent, env.md.engine.global_seed)
    elif kind == "safe_teacher": pol = make_safe_expert(env)
    c = c_auto(*weather, env.table) if arm == "caut_auto" else 0.0; p = keep_prob(*weather, env.table)
    vw, t, info, speeds, t0 = None, 0, None, [], time.time()
    while True:
        raw = (o[:6] * SCALE).astype(np.float32)
        if kind in ("ov", "cond"):
            bev = env.student_bev(); b, s, cc = prep(torch, bev, raw, c, P["mu"], P["sd"])
            with torch.no_grad(): act = P["net"](b, s, cc).squeeze(0).numpy()
        elif kind in ("raw", "abl"):
            bev = env.asm.build_bev_grid([x for x in env.truth_objects() if env.drng.random() < p]); sc = ((raw - P["mu"]) / P["sd"]).astype(np.float32)
            if kind == "abl":
                ov, mask, blank = P["cfg"]
                if ov: bev = overlay_github_bev(bev, env.md)
                if blank: bev = np.zeros_like(bev)
                sc = sc * np.asarray(mask, np.float32)
            with torch.no_grad(): act = P["net"](torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc[None])).squeeze(0).numpy()
        else:
            act = np.clip(np.asarray(pol.act(env) if kind == "safe_teacher" else pol.act(), np.float32)[:2], -1, 1)
        o, r, te, tr, info = env.step(act); t += 1; speeds.append(info["mean_speed"])
        frame = frame_from_sensor(env.md, "front_left_camera")
        if weather[0] != "clear":
            np.random.seed(20261110 + seed * 1000 + t); frame = apply_weather(frame, weather[0], weather[1])
        if vw is None: vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if te or tr or (max_steps and t >= max_steps): break
    vw.release()
    return {"arm": arm, "seed": seed, "weather": f"{weather[0]}:{weather[1]}", "outcome": info["outcome"] or ("cut" if max_steps else "?"), "steps": t,
            "mean_speed": float(np.mean(speeds)), "file": out.name, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/videos_front_camera"); ap.add_argument("--seeds", nargs="+", type=int, default=[1039])
    ap.add_argument("--arms", nargs="+", default=ARMS); ap.add_argument("--weather", nargs="+", default=["fog:0.5", "rain:0.5", "clear:0.0"])
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch
    import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
    from bev_rl.adaptive.record_front_camera import make_env
    env = make_env(); idx = out / "index.jsonl"; ws = [(x.split(":")[0], float(x.split(":")[1])) for x in a.weather]
    for w in ws:
        for arm in a.arms:
            if arm == "caut_auto" and w[0] == "clear": continue                         # identical to caut_c0 in clear weather
            for s in a.seeds:
                f = out / f"{arm}__seed{s}__{w[0]}{'' if w[0] == 'clear' else w[1]}.mp4"
                if a.skip_existing and f.exists(): continue
                try:
                    rec = record(arm, s, w, f, env, torch, a.max_steps); print(json.dumps(rec), flush=True)
                    with open(idx, "a") as fh: fh.write(json.dumps(rec) + "\n")
                except Exception as e: print("FAILED", arm, s, w, repr(e), flush=True)
    env.close()


if __name__ == "__main__":
    main()
