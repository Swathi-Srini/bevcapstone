"""VANILLA camera policy: ONE front camera image -> small CNN -> (steer, throttle). Nothing else goes in (no BEV, no speed, no scalars, no weather label).

Purpose: show that image corruptions (fog, rain, sensor noise, blur, darkness, occlusion) really HURT a policy that sees raw pixels - the control experiment for the BEV models, whose
input is a symbolic grid that these corruptions barely touch.

Pipeline (all new code, policy side only; the teammates' camera/weather modules are imported read-only):
  render  : MetaDrive RGBCamera 320x240, FOV 60, mounted on the teammates' front_left_camera rig (re-mounted after EVERY reset)
  corrupt : frame (BGR 320x240) -> corruption(kind, level)   [fog/rain = teammates' apply_weather; noise/blur/dark/occlusion = defined here]
  net     : resize to 128x96 RGB, /255 -> 5 conv layers -> MLP -> tanh(2)
Training = behaviour cloning of the privileged SafeTeacher (ground-truth state) on CLEAN images only, with action-noise injection + DAgger rounds (CNN drives, teacher relabels).
    python -m bev_rl.adaptive.vanilla_camera_cnn train  --name vcam --episodes0 40 --dagger-iters 3 --dagger-episodes 20
    python -m bev_rl.adaptive.vanilla_camera_cnn eval   --name vcam --n 40 --seed0 5000 --conds clear fog:1.0 rain:1.0 noise:0.5 ...
    python -m bev_rl.adaptive.vanilla_camera_cnn frames --out paper/vanilla_camera/corruption_examples.png
Condition string  kind:level  with level in [0,1]:  clear | fog | rain | noise | blur | dark | occlude
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2
import numpy as np

RENDER_W, RENDER_H = 320, 240
NET_W, NET_H = 128, 96
RUNS = Path("bev_rl/adaptive/runs")
OUT = Path("bev_rl/adaptive/results/vanilla_camera")


# ----------------------------------------------------------------------------------------------------------------- corruptions
def corrupt(frame, kind, level, rng):
    """frame: uint8 BGR HxWx3. level in [0,1]. Returns a corrupted copy (same shape/dtype)."""
    if kind == "clear" or level <= 0: return frame
    h, w = frame.shape[:2]
    if kind in ("fog", "rain"):
        from weather.weather_utils import apply_weather                      # teammates' augmentation, unmodified (uses np.random global state -> reseed per frame)
        np.random.seed(int(rng.integers(0, 2 ** 31 - 1))); return apply_weather(frame, kind, level)
    if kind == "noise":                                                        # additive gaussian sensor noise, sigma = 80*level grey levels
        return np.clip(frame.astype(np.float32) + rng.normal(0, 80.0 * level, frame.shape), 0, 255).astype(np.uint8)
    if kind == "blur":                                                         # defocus / lens smear: odd gaussian kernel up to 31 px
        k = int(2 * round(15 * level) + 1); return cv2.GaussianBlur(frame, (k, k), 0)
    if kind == "dark":                                                         # low light: multiply by (1 - 0.9*level)
        return np.clip(frame.astype(np.float32) * (1.0 - 0.9 * level), 0, 255).astype(np.uint8)
    if kind == "occlude":                                                      # opaque grey patch (mud / sticker), area fraction = level*0.6, random place per EPISODE-frame
        out = frame.copy(); ph, pw = int(h * np.sqrt(0.6 * level)), int(w * np.sqrt(0.6 * level))
        y0, x0 = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1)); out[y0:y0 + ph, x0:x0 + pw] = 128; return out
    raise ValueError(kind)


def parse_cond(s):
    k, _, l = s.partition(":"); return (k, float(l) if l else 0.0)


def cname(c): return "clear" if c[0] == "clear" else f"{c[0]}{c[1]:g}"


# ----------------------------------------------------------------------------------------------------------------- env with a small camera
def make_cam_env(horizon=800, density=0.05, seeds=range(0, 10)):
    """DegradedEnv (perfect perception, mode='none') whose MetaDrive instance is replaced by one with a 320x240 front camera (used ONLY as the policy's input here)."""
    import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from bev_rl.adaptive.degraded_env import DegradedEnv

    class CamEnv(DegradedEnv):
        def __init__(self):
            super().__init__(list(seeds), [("clear", 0.0)], mode="none", traffic_density=density, horizon=horizon, sample_seed=7)
            self.md.close()
            self.md = MetaDriveEnv({"use_render": False, "image_observation": True, "norm_pixel": False, "traffic_density": float(density), "num_scenarios": 10000, "start_seed": 10,
                                    "horizon": horizon, "random_agent_model": False, "random_lane_width": True, "random_lane_num": True, "out_of_route_done": True,
                                    "on_continuous_line_done": False, "manual_control": False, "log_level": 50,
                                    "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False, "show_line_to_navi_mark": False, "show_navigation_arrow": False},
                                    "sensors": {"front_left_camera": (RGBCamera, RENDER_W, RENDER_H)}})

        def reset(self, *, seed=None, options=None):
            o, i = super().reset(seed=seed, options=options)
            from manual_drive_stereo_yolo_weather import CAMERA_RIGS, CAM_FOV
            pos, hpr = CAMERA_RIGS["front_left_camera"]; s = self.md.engine.get_sensor("front_left_camera")
            s.track(self.md.agent.origin, pos, hpr); s.get_lens().setFov(CAM_FOV)       # remount after EVERY reset (mount-once leaves a stale camera)
            return o, i

        def frame(self):
            from manual_drive_stereo_yolo_weather import frame_from_sensor
            return frame_from_sensor(self.md, "front_left_camera")                       # BGR uint8 320x240
    return CamEnv()


def to_net_input(frame_bgr):
    """BGR 320x240 uint8 -> RGB 128x96 uint8, CHW (stored as uint8; the net divides by 255)."""
    return np.ascontiguousarray(cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (NET_W, NET_H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1))


# ----------------------------------------------------------------------------------------------------------------- network
def build_net():
    import torch.nn as nn

    class VanillaCNN(nn.Module):
        def __init__(self):
            super().__init__()
            c = lambda i, o, k, s: [nn.Conv2d(i, o, k, s, k // 2), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            self.conv = nn.Sequential(*c(3, 24, 5, 2), *c(24, 36, 5, 2), *c(36, 48, 3, 2), *c(48, 64, 3, 2), *c(64, 64, 3, 2))        # 96x128 -> 3x4
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3 * 4, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, 2), nn.Tanh())

        def forward(self, x):                                                          # x: uint8-valued float (B,3,96,128) in 0..255
            return self.head(self.conv(x / 255.0))
    return VanillaCNN()


def act_net(net, torch, img):
    with torch.no_grad(): return net(torch.from_numpy(img[None].astype(np.float32))).squeeze(0).numpy()


# ----------------------------------------------------------------------------------------------------------------- one episode
def episode(env, torch, seed, cond=("clear", 0.0), net=None, beta=0.0, noise=0.0, collect=False, rng=None):
    """net=None -> teacher drives. collect -> returns (imgs, expert_actions, info). Corruption is applied to every frame the NET sees; the teacher always uses ground truth."""
    from bev_rl.adaptive.degraded_env import make_safe_expert
    o, _ = env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); env.drng = np.random.default_rng(seed * 7 + 1)
    crng = np.random.default_rng(seed * 13 + 5); rng = rng if rng is not None else np.random.default_rng(seed)
    pol = make_safe_expert(env) if (collect or net is None) else None
    X, Y, gap = [], [], 30.0
    while True:
        img = to_net_input(corrupt(env.frame(), cond[0], cond[1], crng))
        ea = np.clip(np.asarray(pol.act(env), np.float32)[:2], -1, 1) if pol is not None else None
        if collect: X.append(img); Y.append(ea)
        a = ea if (net is None or (collect and rng.random() < beta)) else act_net(net, torch, img)
        if noise > 0: a = np.clip(a + rng.normal(0, noise, 2), -1, 1).astype(np.float32)
        o, r, te, tr, info = env.step(a); gap = min(gap, env.min_gap_m())
        if te or tr: info["min_gap_m"] = gap; return X, Y, info


# ----------------------------------------------------------------------------------------------------------------- train (BC + DAgger)
def fit(net, X, Y, torch, epochs, lr, dev="cpu"):
    X = torch.from_numpy(np.stack(X)); Y = torch.from_numpy(np.stack(Y)); n = len(X); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    perm = torch.randperm(n); nv = max(200, n // 20); vi, ti = perm[:nv], perm[nv:]; best, hist = (1e9, None), []
    for ep in range(epochs):
        net.train(); p = ti[torch.randperm(len(ti))]; tl = 0.0
        for i in range(0, len(p), 128):
            b = p[i:i + 128]; x = X[b].float()
            x = x * (0.8 + 0.4 * torch.rand(len(b), 1, 1, 1)) + 12 * torch.randn(len(b), 1, 1, 1)           # mild brightness/contrast jitter only (the net is trained on CLEAN-style images)
            loss = torch.nn.functional.smooth_l1_loss(net(x.clamp(0, 255)), Y[b], beta=0.1); opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(b)
        net.eval()
        with torch.no_grad(): vl = float(np.mean([torch.nn.functional.smooth_l1_loss(net(X[vi[i:i + 256]].float()), Y[vi[i:i + 256]], beta=0.1).item() for i in range(0, nv, 256)]))
        hist.append((tl / len(ti), vl)); print(f"  epoch {ep + 1}/{epochs} train {tl / len(ti):.4f} val {vl:.4f}", flush=True)
    return hist


def cmd_train(a):
    import torch
    torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
    run = RUNS / a.name; run.mkdir(parents=True, exist_ok=True); env = make_cam_env(seeds=range(100, 200)); net = build_net(); X, Y, log = [], [], []
    seeds = list(range(100, 200)); rng.shuffle(seeds); k = 0
    t0 = time.time()
    for ep in range(a.episodes0):                                                  # round 0: teacher drives with action-noise injection (0, 0.1, 0.25) so the data covers recoveries
        nz = [0.0, 0.1, 0.25][ep % 3]; x, y, info = episode(env, torch, seeds[k % 100], net=None, noise=nz, collect=True, rng=rng); k += 1
        X += x; Y += y; print(f"[bc0 ep{ep}] noise {nz} {info['outcome']} frames={len(x)} total={len(X)} t={time.time() - t0:.0f}s", flush=True)
    fit(net, X, Y, torch, a.epochs, 1e-3)
    for it in range(a.dagger_iters):                                               # DAgger: the CNN drives (beta-mixed with teacher), teacher relabels every frame
        beta = 0.3 * (1 - it / max(1, a.dagger_iters)); outs = []
        net.eval()
        for ep in range(a.dagger_episodes):
            x, y, info = episode(env, torch, seeds[k % 100], net=net, beta=beta, collect=True, rng=rng); k += 1; X += x; Y += y; outs.append(info["outcome"])
        print(f"[dagger {it + 1}] beta {beta:.2f} outcomes {dict((o, outs.count(o)) for o in set(outs))} total frames {len(X)} t={time.time() - t0:.0f}s", flush=True)
        log.append({"iter": it + 1, "beta": beta, "outcomes": outs, "frames": len(X)})
        fit(net, X, Y, torch, a.epochs, 5e-4)
        torch.save({"model_state": net.state_dict(), "frames": len(X)}, run / "best.pt")
    torch.save({"model_state": net.state_dict(), "frames": len(X)}, run / "best.pt"); (run / "log.json").write_text(json.dumps({"args": vars(a), "dagger": log, "frames": len(X)}, indent=1))
    env.close()


# ----------------------------------------------------------------------------------------------------------------- eval
def cmd_eval(a):
    import torch
    torch.set_num_threads(2); ck = torch.load(RUNS / a.name / "best.pt", map_location="cpu", weights_only=True); net = build_net(); net.load_state_dict(ck["model_state"]); net.eval()
    OUT.mkdir(parents=True, exist_ok=True); env = make_cam_env(seeds=range(a.seed0, a.seed0 + a.n))
    for cs in a.conds:
        c = parse_cond(cs); f = OUT / f"{a.name}__{cname(c)}__n{a.n}_s{a.seed0}.json"; part = f.with_suffix(".partial.json")
        if f.exists(): continue
        rows = json.loads(part.read_text()) if part.exists() else []; done = {r["seed"] for r in rows}
        for s in range(a.seed0, a.seed0 + a.n):
            if s in done: continue
            _, _, i = episode(env, torch, s, cond=c, net=net)
            rows.append({"seed": s, "outcome": i["outcome"], "speed": float(i["mean_speed"]), "route": float(i["route_completion"]), "min_gap_m": float(i["min_gap_m"]),
                         "wh_per_km": float(i["energy_wh"] / max(i["distance_m"] / 1000.0, 1e-3)), "steps": int(i["steps"])}); print(cname(c), rows[-1], flush=True)
            if len(rows) % 10 == 0: part.write_text(json.dumps(rows))
        oc = [r["outcome"] for r in rows]
        f.write_text(json.dumps({"model": a.name, "cond": cname(c), "n": len(rows), **{k: oc.count(k) / len(rows) for k in ("success", "crash", "off_road", "timeout")},
                                 "mean_speed": float(np.mean([r["speed"] for r in rows])), "route": float(np.mean([r["route"] for r in rows])), "episodes": rows}, indent=1))
        part.unlink(missing_ok=True); print("SUMMARY", cname(c), f"success {oc.count('success') / len(oc):.3f} n={len(oc)}", flush=True)
    env.close()


# ----------------------------------------------------------------------------------------------------------------- picture of every corruption
def cmd_frames(a):
    env = make_cam_env(seeds=[5001]); env.reset(options={"scenario_seed": 5001, "weather": ("clear", 0.0)})
    for _ in range(60): env.step(np.array([0.0, 0.5], np.float32))
    fr = env.frame(); rng = np.random.default_rng(0); tiles = []
    for cs in a.conds:
        c = parse_cond(cs); im = corrupt(fr, c[0], c[1], rng).copy(); cv2.putText(im, cname(c), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(cv2.resize(im, (RENDER_W, RENDER_H)))
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]); Path(a.out).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(a.out, grid); print("wrote", a.out)
    env.close()


DEFAULT_CONDS = ["clear", "fog:0.5", "fog:1.0", "rain:0.5", "rain:1.0", "noise:0.5", "noise:1.0", "blur:0.5", "blur:1.0", "dark:0.5", "dark:0.9", "occlude:0.5", "occlude:1.0"]


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train"); t.add_argument("--name", default="vcam"); t.add_argument("--episodes0", type=int, default=40); t.add_argument("--dagger-iters", type=int, default=3)
    t.add_argument("--dagger-episodes", type=int, default=20); t.add_argument("--epochs", type=int, default=8); t.set_defaults(f=cmd_train)
    e = sp.add_parser("eval"); e.add_argument("--name", default="vcam"); e.add_argument("--n", type=int, default=40); e.add_argument("--seed0", type=int, default=5000)
    e.add_argument("--conds", nargs="+", default=DEFAULT_CONDS); e.set_defaults(f=cmd_eval)
    f = sp.add_parser("frames"); f.add_argument("--out", default="paper/vanilla_camera/corruption_examples.png"); f.add_argument("--conds", nargs="+", default=DEFAULT_CONDS[:12]); f.set_defaults(f=cmd_frames)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
