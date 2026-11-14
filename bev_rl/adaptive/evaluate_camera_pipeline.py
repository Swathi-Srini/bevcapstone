"""FULL CAMERA-PIPELINE EVALUATION of policies (matched seeds, identical conditions for every model), with optional front-camera videos.

Every step: 5 cameras -> weather filter -> front stereo depth + YOLOv8n (4 views) -> unmodified objects_from_detections -> frozen CameraGridBuilder (+ HD-map layers) -> 64x64 BEV.
Policy input: that BEV + ego scalars (mask of the checkpoint) + caution scalar for caution models. No teacher, no action noise: the policy drives every step.
Simulator ground truth is used ONLY to score perception (recall / BEV agreement), never as input.

Arms
    ov_split_original   runs/ov_split/best.pt   ORIGINAL checkpoint trained on simulator-list grids ("200-episode model"), no caution input
    caution_original    runs/caut_a/best.pt     ORIGINAL caution checkpoint (200-episode model); caution c = 1 - keep_prob(weather) from the ORIGINAL retention table, as it was trained
    ov_camera_final     camera_dagger_final/e2e_ov_split_final/best.pt   (camera-trained OV-Split student, after its training finishes)
Tasks    --tasks "fog:1.0@5100-5109;clear:0.0@5100;rain:1.0@5100"     (weather@seed range, ';'-separated)
Videos   --video-seeds 5100   records a front-camera + BEV video for every episode whose seed is listed (out/videos/<arm>__<weather>__s<seed>.mp4, 10 fps)
Settings density 0.05, horizon 800 steps (80 s). Resumable: one claim file per episode, one result file per episode. Refuses to run if the frozen files changed.

    python -m bev_rl.adaptive.evaluate_camera_pipeline --worker 1 --out bev_rl/adaptive/results/camera_eval_originals --tasks "fog:1.0@5100-5109" --video-seeds 5100
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
ARMS = {"ov_split_original": ("ov_split", False), "caution_original": ("caut_auto", True), "ov_camera_final": ("CKPT:bev_rl/adaptive/results/camera_dagger_final/e2e_ov_split_final/best.pt", False)}
MAX_WORKERS = int(os.environ.get("CAM_MAX_WORKERS", 3))
# BEV colours (BGR) for the video: free, centreline 0.5, boundary 0.8, vehicle 1.0
PALETTE = {0.0: (40, 40, 40), 0.5: (0, 215, 255), 0.8: (200, 120, 0), 1.0: (0, 0, 255)}


def parse_tasks(s):
    out = []
    for part in s.split(";"):
        w, seeds = part.split("@"); kind, level = w.split(":"); lo, hi = (seeds.split("-") + [seeds])[:2]
        out += [((kind, float(level)), seed) for seed in range(int(lo), int(hi) + 1)]
    return out


def load_arm(arm, torch):
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.evaluate_n200_all_weather import load_model
    spec = ARMS[arm][0]
    if not spec.startswith("CKPT:"): return load_model(spec, torch)
    ck = torch.load(spec[5:], map_location="cpu", weights_only=True); inner = BEVScalarPolicy(1, 6, 0); inner.load_state_dict(ck["model_state"]); inner.eval()
    return (lambda b, s_, c: inner(b, s_)), np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def video_frame(front, grid, arm, weather, step, speed, c):
    import cv2
    f = cv2.resize(front, (640, 480)); g = np.zeros((64, 64, 3), np.uint8)
    for v, col in PALETTE.items(): g[np.isclose(grid, v, atol=0.05)] = col
    g = cv2.resize(g, (480, 480), interpolation=cv2.INTER_NEAREST); cv2.rectangle(g, (int(32 * 7.5) - 6, int(56 * 7.5) - 9), (int(32 * 7.5) + 6, int(56 * 7.5) + 9), (0, 255, 0), 1)
    img = np.concatenate([f, g], axis=1)
    cv2.putText(img, f"{arm}  {weather}  step {step}  v={speed:.1f} m/s  c={c:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "front camera (weathered)", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); cv2.putText(img, "camera-derived BEV (red = detected vehicle)", (650, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--worker", type=int, default=1)
    ap.add_argument("--arms", default="ov_split_original,caution_original"); ap.add_argument("--video-seeds", default=""); ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--horizon", type=int, default=800)
    a = ap.parse_args()
    if a.worker > MAX_WORKERS: return
    from bev_rl.adaptive.camera_dagger import verify_frozen
    verify_frozen()
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import prep, c_auto, RET_PATH
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.evaluate_n200_all_weather import row
    from bev_rl.adaptive.pilot_camera_builder import PilotEnv
    arms = a.arms.split(","); vseeds = {int(x) for x in a.video_seeds.split(",") if x}; table = json.loads(RET_PATH.read_text())
    for d in ("episodes", "claims", "videos"): (a.out / d).mkdir(parents=True, exist_ok=True)
    pe = PilotEnv(density=a.density, horizon=a.horizon); env = pe.env; nets = {k: load_arm(k, torch) for k in arms}
    for (kind, level), seed in parse_tasks(a.tasks):
        for arm in arms:
            name = f"{arm}__{kind}_{level}__s{seed}"; res = a.out / "episodes" / f"{name}.json"
            if res.exists(): continue
            try: fd = os.open(a.out / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
            except FileExistsError: continue
            with open(a.out / "claims" / f"_w{a.worker}.log", "a") as fh: fh.write(name + "\n")                          # lets a gate release THIS worker's unfinished claims safely
            net, mu, sd = nets[arm]; uses_c = ARMS[arm][1]; c = c_auto(kind, level, table) if uses_c else 0.0; t0 = time.time(); vw = None
            o, _ = env.reset(options={"scenario_seed": seed, "weather": (kind, level)}); env.drng = np.random.default_rng(seed * 7 + 1); steps = []
            if seed in vseeds:
                import cv2
                vw = cv2.VideoWriter(str(a.out / "videos" / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1120, 480))
            while True:
                grid, rec, _ = pe.grid_and_metrics(); steps.append(rec)
                raw = (o[:6] * SCALE).astype(np.float32)
                with torch.no_grad(): act = net(*prep(torch, grid, raw, c, mu, sd)).squeeze(0).numpy()
                if vw is not None: vw.write(video_frame(pe.last_front, grid, arm, f"{kind} {level}", len(steps), float(env.md.agent.speed), c))
                o, r, te, tr, info = env.step(act)
                if te or tr: break
            if vw is not None: vw.release()
            out = row(seed, info); out.update({"arm": arm, "weather": f"{kind}:{level}", "c_input": c, "density": a.density, "horizon": a.horizon, "wall_s": round(time.time() - t0), "video": vw is not None, "steps_log": steps})
            res.write_text(json.dumps(out)); print(json.dumps({k: v for k, v in out.items() if k != "steps_log"}), flush=True)
    env.md.close()


if __name__ == "__main__":
    main()
