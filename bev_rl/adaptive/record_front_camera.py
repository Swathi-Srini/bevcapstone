"""FRONT-CAMERA recordings of every model on ONE seed under different weather (clear / fog 0.5 / rain 0.5 / ...).

What you see per frame (960x480):
  LEFT   the teammates' front-left camera frame (1200x900 render) with the GitHub weather augmentation applied (apply_weather, same level as the policy's detector dropout) and the
         YOLO detections the teammates' detector produces on THAT frame (green boxes). This is display/perception context.
  RIGHT  the BEV the policy actually receives (bright colours: unknown grey, free black, ego blue, vehicles red, route green, boundary yellow) PLUS a DISPLAY-ONLY layer of the full lane geometry
         from the simulator map (dim lines) so the road is readable. Dim lines are NOT model input.
IMPORTANT: the policies never see camera pixels (the camera pipeline is not integrated with the policy). They drive from the simulator-derived BEV with weather = detector dropout
(measured retention table); the camera + YOLO panel shows what the real perception would be looking at under the same weather level.
Scenario = the evaluation environment (same MetaDrive config, density 0.05, horizon 800) + one front camera. 10 Hz sim; one frame every --every steps (video fps = 10/every).
    python -m bev_rl.adaptive.record_front_camera --seeds 1039 --weather clear:0.0 fog:0.5 rain:0.5 --out paper/videos_camera
File: <out>/<model>__seed<seed>__<weather>.mp4  (weather tag: clear, fog0.5, rain0.5 ...)
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2
import numpy as np

ARMS = ["repo_bc_sparse", "repo_bc_clear", "base_a", "ov_split", "ctrl_nocaut", "caut_c0", "caut_auto", "safe_student", "stock_idm", "safe_teacher",
        "arch_bev_only", "arch_scalar_only", "ov_both", "ov_bev_only"]
DIM = {"edge": (170, 130, 70), "route": (50, 120, 60)}            # BGR: display-only lane edges steel blue (distinct from the bright yellow model-input boundary), dim green route


def make_md_cam(horizon, density):
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
    from manual_drive_stereo_yolo_weather import CAM_H, CAM_W
    return MetaDriveEnv({"use_render": False, "image_observation": True, "norm_pixel": False, "traffic_density": float(density), "num_scenarios": 10000, "start_seed": 10,
                         "horizon": horizon, "random_agent_model": False, "random_lane_width": True, "random_lane_num": True, "out_of_route_done": True,
                         "on_continuous_line_done": False, "manual_control": False, "log_level": 50,
                         "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False, "show_line_to_navi_mark": False, "show_navigation_arrow": False},
                         "sensors": {"front_left_camera": (RGBCamera, CAM_W, CAM_H)}})


def display_layers(md):
    """Display-only: lane edges of ALL parallel lanes of the current road (+ lookahead) and the ego route centreline, in the ego-frame BEV grid."""
    from bev_rl.adaptive.spec_bev import GRID, _to_cells
    ag = md.agent; pos = np.asarray(ag.position, float); th = float(ag.heading_theta)
    fwd = np.array([np.cos(th), np.sin(th)]); right = np.array([np.sin(th), -np.cos(th)])
    nav = ag.navigation; edge = np.zeros((GRID, GRID), bool); route = np.zeros((GRID, GRID), bool)
    lanes = list(getattr(nav, "current_ref_lanes", None) or [nav.current_lane]) + list(getattr(nav, "next_ref_lanes", None) or [])
    for lane in lanes:
        try:
            s0, _ = lane.local_coordinates(ag.position); s = np.arange(max(0.0, s0 - 4.0), lane.length, 0.2)
            if s.size == 0: continue
            w = np.array([lane.width_at(float(x)) if hasattr(lane, "width_at") else lane.width for x in s])
            for sc, tgt in ((0.5, edge), (-0.5, edge), (0.0, route if lane is nav.current_lane else None)):
                if tgt is None: continue
                pts = np.array([lane.position(float(si), float(wi * sc)) for si, wi in zip(s, w)]); d = pts - pos
                r, c = _to_cells(d @ right, d @ fwd); tgt[r, c] = True
        except Exception:
            continue
    return edge, route


def bev_vis(bev, edge, route, size=288):
    from bev_rl.adaptive.record_videos import PAL
    keys = np.array(list(PAL)); cols = np.array([PAL[k] for k in PAL], np.uint8)
    img = cols[np.abs(bev[..., None] - keys[None, None, :]).argmin(-1)][..., ::-1].astype(np.uint8)
    free = (bev == 0.0) | (bev == -1.0)
    img[edge & free] = DIM["edge"]; img[route & free] = DIM["route"]                 # display-only layer, drawn only where the input has nothing
    ego_rows = np.where(np.abs(bev - 0.9) < 0.05)[0]
    if len(ego_rows) and ego_rows.mean() < bev.shape[0] / 2: img = img[::-1]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def compose(cam, bevimg, lines, act, W=960, H=480):
    f = np.full((H, W, 3), 245, np.uint8); f[:, :640] = cv2.resize(cam, (640, 480))
    f[28:28 + 288, 656:656 + 288] = bevimg
    cv2.putText(f, "policy BEV input (forward = up)", (656, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(f, "bright = model input | dim lines = display only", (656, 326), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    for k, t in enumerate(lines): cv2.putText(f, t, (656, 350 + 19 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5 if k else 0.58, (20, 20, 20), 1 if k else 2, cv2.LINE_AA)
    for j, (name, v) in enumerate((("steer", act[0]), ("throttle(+)/brake(-)", act[1]))):
        yy = 440 + 24 * j; cv2.putText(f, name, (656, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.rectangle(f, (776, yy - 11), (944, yy + 1), (200, 200, 200), 1)
        cv2.rectangle(f, (860, yy - 10), (int(860 + 83 * float(np.clip(v, -1, 1))), yy), (40, 120, 220) if v >= 0 else (60, 60, 200), -1)
    return f


def make_env(horizon=800, density=0.05):
    from bev_rl.adaptive.degraded_env import DegradedEnv

    class CamEnv(DegradedEnv):
        def __init__(self):
            super().__init__([1039], [("clear", 0.0)], mode="dropout", traffic_density=density, horizon=horizon, sample_seed=7)
            self.md.close(); self.md = make_md_cam(horizon, density)

        def reset(self, *, seed=None, options=None):
            o, i = super().reset(seed=seed, options=options)
            from manual_drive_stereo_yolo_weather import CAMERA_RIGS, CAM_FOV
            pos, hpr = CAMERA_RIGS["front_left_camera"]; s = self.md.engine.get_sensor("front_left_camera")
            s.track(self.md.agent.origin, pos, hpr); s.get_lens().setFov(CAM_FOV)             # remount after EVERY reset (mount-once would leave a stale camera)
            return o, i
    return CamEnv()


def record(arm, seed, weather, out, env, torch, yolo, every, max_steps):
    from bev_rl.adaptive.caution import c_auto, keep_prob, prep
    from bev_rl.adaptive.degraded_env import make_safe_expert
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.record_videos import ABL, OUTCOME_COL, build
    from bev_rl.adaptive.spec_bev import overlay_github_bev
    from manual_drive_stereo_yolo_weather import frame_from_sensor
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import annotate_image, run_yolo
    P = build(arm, torch, env); kind = P["kind"]
    o, _ = env.reset(options={"scenario_seed": seed, "weather": weather}); env.drng = np.random.default_rng(seed * 7 + 1)
    pol = None
    if kind == "stock_idm":
        from metadrive.policy.idm_policy import IDMPolicy
        pol = IDMPolicy(env.md.agent, env.md.engine.global_seed)
    elif kind == "safe_teacher": pol = make_safe_expert(env)
    c = c_auto(*weather, env.table) if arm == "caut_auto" else 0.0; p = keep_prob(*weather, env.table)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0 / every, (960, 480)); t, info, speeds, tcost = 0, None, [], 0.0
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
            act = np.clip(np.asarray(pol.act(env) if kind == "safe_teacher" else pol.act(), np.float32)[:2], -1, 1); bev = env.student_bev()
        o, r, te, tr, info = env.step(act); t += 1; speeds.append(info["mean_speed"]); end = bool(te or tr or (max_steps and t >= max_steps))
        if t % every == 0 or end:
            t0 = time.time(); cam = frame_from_sensor(env.md, "front_left_camera")
            if weather[0] != "clear":
                np.random.seed(20261110 + seed * 1000 + t); cam = apply_weather(cam, weather[0], weather[1])
            dets = run_yolo(yolo, cam); cam = annotate_image(cam, dets); edge, route = display_layers(env.md); tcost += time.time() - t0
            wtxt = f"seed {seed}   {weather[0]} {weather[1]}" + (f"   c={c:.2f}" if arm == "caut_auto" else "")
            lines = [arm, wtxt, f"t = {t / 10:5.1f} s   speed {float(env.md.agent.speed):4.1f} m/s", f"YOLO boxes {len(dets)}   vehicles in BEV window {len(env.truth_objects())}",
                     "rule-based: sees ground truth" if kind in ("stock_idm", "safe_teacher") else "policy input = BEV only (no pixels)"]
            last = compose(cam, bev_vis(bev, edge, route), lines, act); vw.write(last)
        if end: break
    outcome = info["outcome"] or ("cut" if max_steps else "?")
    for _ in range(max(4, 18 // every)):
        fr = last.copy(); col = OUTCOME_COL.get(outcome, (90, 90, 90)); cv2.rectangle(fr, (60, 190), (580, 290), (255, 255, 255), -1); cv2.rectangle(fr, (60, 190), (580, 290), col, 4)
        cv2.putText(fr, outcome.upper().replace("_", "-"), (90, 262), cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 4, cv2.LINE_AA); vw.write(fr)
    vw.release()
    return {"arm": arm, "seed": seed, "weather": f"{weather[0]}:{weather[1]}", "outcome": outcome, "steps": t, "mean_speed": float(np.mean(speeds)), "file": out.name, "perception_s_per_frame": tcost / max(1, t // every)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/videos_camera"); ap.add_argument("--seeds", nargs="+", type=int, default=[1039])
    ap.add_argument("--arms", nargs="+", default=ARMS); ap.add_argument("--weather", nargs="+", default=["clear:0.0", "fog:0.5", "rain:0.5"])
    ap.add_argument("--every", type=int, default=2); ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import torch, ultralytics  # noqa
    import bev_state  # noqa
    from yolo.yolo_utils import ensure_yolo_model
    yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, 0.4); env = make_env(); idx = out / "index.jsonl"
    ws = [(x.split(":")[0], float(x.split(":")[1])) for x in a.weather]
    for w in ws:
        for arm in a.arms:
            if arm == "caut_auto" and w[0] == "clear": continue                          # identical to caut_c0 in clear weather
            for s in a.seeds:
                f = out / f"{arm}__seed{s}__{w[0]}{'' if w[0] == 'clear' else w[1]}.mp4"
                if a.skip_existing and f.exists(): continue
                try:
                    rec = record(arm, s, w, f, env, torch, yolo, a.every, a.max_steps); print(json.dumps(rec), flush=True)
                    with open(idx, "a") as fh: fh.write(json.dumps(rec) + "\n")
                except Exception as e: print("FAILED", arm, s, w, repr(e), flush=True)
    env.close()


if __name__ == "__main__":
    main()
