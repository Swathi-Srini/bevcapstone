"""Camera-pipeline DAgger for the two policies, as SEPARATE loops with separate datasets and checkpoints.

    model ov    OV-Split Camera      init runs/ov_split/best.pt     teacher SafeTeacher + curve guard       weather: clear only (as the original)
    model caut  Live Caution Camera  init runs/caut_a/best.pt       teacher caution-SafeTeacher + guard     weather: original 8-item mix, caution c = U(0,1) 50% / c_auto(weather) 50%

Student observation in EVERY rollout and validation episode: cameras -> GitHub weather filter -> YOLOv8n (4 views) + front stereo depth -> unmodified objects_from_detections
-> CameraGridBuilder (+ HD-map layers) -> 64x64 grid, plus the ego scalars (speed, progress, goal distance) from simulator ego state, plus caution c for the caution model.
The privileged teacher reads the exact simulator state and is used ONLY to label the visited state. No simulator vehicle position reaches the student.

Recipe (original, except the teachers and the lean round count): per round 40 episodes = 8 at each density 0 / 0.05 / 0.1 / 0.2 / 0.3, training seeds 2000-2999, horizon 600, action noise 0.1,
beta = 1 (teacher drives) in round 0 and 0.5^t later, retraining on ALL data so far with Adam lr 1e-3, batch 128, MSE, epochs 24/8 (ov) or 18/6 (caut). The network persists across rounds
(initialised from the original checkpoint, whose scalar normalisation is kept). Selection: validation seeds 3100-3104 at density 0.05 (ov: clear + rain 1.0, horizon 800; caut: clear + rain 0.6 +
rain 1.0 with caution c_auto, horizon 1600), score = (mean success, -mean(crash+off_road)) as in the original. ORIGINAL CHECKPOINTS ARE NEVER WRITTEN.

    python -m bev_rl.adaptive.camera_dagger collect  --model ov --round 0 --worker 1
    python -m bev_rl.adaptive.camera_dagger train    --model ov --round 0
    python -m bev_rl.adaptive.camera_dagger validate --model ov --round 0 --worker 1
    python -m bev_rl.adaptive.camera_dagger select   --model ov --round 0
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
DENSITIES = [0.05]                                   # APPROVED 2026-11-14: density 0.05 exclusively (training, validation, evaluation)
TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.6), ("fog", 1.0), ("rain", 0.2), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3110))
ROOT = Path("bev_rl/adaptive/results/camera_dagger_final")          # new directory; the earlier camera_dagger/ outputs are archived in place, never used here
MODELS = {
    "ov": dict(init="bev_rl/adaptive/runs/ov_split/best.pt", name="e2e_ov_split_final", teacher="safe_g4.0", weather=[("clear", 0.0)], uses_c=False, epochs0=24, epochs=8,
               val_weather=[("clear", 0.0), ("rain", 1.0)], val_horizon=800),
    "caut": dict(init="bev_rl/adaptive/runs/caut_a/best.pt", name="e2e_caut_final", teacher="cs0_g4.0", weather=TRAIN_WEATHER, uses_c=True, epochs0=18, epochs=6,
                 val_weather=[("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)], val_horizon=800),
}


def make_env(density, horizon, tag, sensor=False):
    from bev_rl.adaptive.e2e_pipeline import E2EEnv
    from bev_state.camera_grid_builder import CameraGridBuilder
    import bev_state.assembler as A

    class BuilderEnv(E2EEnv):
        def student_bev(self, light=False):
            """light=True (teacher-driven step that is not recorded): only advance the weather noise clock and the live caution sensor, skip depth + YOLO + builder."""
            from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
            from weather.weather_utils import apply_weather
            if light:
                kind, level = self.weather; fl, fr_ = frame_from_sensor(self.md, "front_left_camera"), frame_from_sensor(self.md, "front_right_camera")
                if kind != "clear" and level > 0:
                    np.random.seed(20261114 + int(self.md.engine.global_seed) * 100003 + self.step_count); apply_weather(fl, kind, level); fl, _ = apply_synchronized_stereo_weather(fl, fr_, kind, level)
                self.step_count += 1
                if self.sensor is not None: self.c_hat = self.sensor.update(fl)
                return None
            raw = {n: frame_from_sensor(self.md, n) for n in CAMERA_RIGS}
            kind, level = self.weather
            if kind == "clear" or level <= 0: frames = raw
            else:
                np.random.seed(20261114 + int(self.md.engine.global_seed) * 100003 + self.step_count)
                frames = {n: apply_weather(f, kind, level) for n, f in raw.items()}
                frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw["front_left_camera"], raw["front_right_camera"], kind, level)
            self.step_count += 1
            if getattr(self, "sensor", None) is not None: self.c_hat = self.sensor.update(frames["front_left_camera"])      # LIVE caution: from the picture only
            depth = A.front_stereo_depth(frames["front_left_camera"], frames["front_right_camera"], self.asm.matcher)
            dets = {n: A.run_yolo(self.yolo, frames[n]) for n in A.YOLO_CAMERAS}
            return self.builder.build(self.asm.objects_from_detections(dets, depth), self.md)
    env = BuilderEnv(list(range(2000, 3000)), [("clear", 0.0)], traffic_density=density, horizon=horizon, sample_seed=tag)
    env.builder = CameraGridBuilder(env.asm)
    env.sensor = None; env.c_hat = 0.0
    if sensor:
        from bev_rl.adaptive.visibility_sensor import VisibilitySensor
        env.sensor = VisibilitySensor()
    return env


def load_net(path, uses_c, torch):
    from bev_policy.model import BEVScalarPolicy
    ck = torch.load(path, map_location="cpu", weights_only=True); net = BEVScalarPolicy(1, 6, 1 if uses_c else 0); net.load_state_dict(ck["model_state"]); net.eval()
    return net, np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def student_act(net, uses_c, torch, bev, raw, c, mu, sd):
    from bev_rl.adaptive.caution import prep
    b, s, cc = prep(torch, bev, raw, c, mu, sd)
    with torch.no_grad(): return (net(b, s, cc) if uses_c else net(b, s)).squeeze(0).numpy()


def teacher_driver(name, env, c):
    """-> act(env) for the chosen privileged teacher (see teacher_eval.py for the definitions)."""
    from bev_rl.adaptive.teacher_eval import build_teacher, curvature_cap_mps
    idm, safe, target, guard = build_teacher(name, env, c); v_cmd = idm.NORMAL_SPEED

    def act(e):
        if guard is not None:
            cap = curvature_cap_mps(e, float(e.md.agent.speed), guard); idm.NORMAL_SPEED = idm.target_speed = min(v_cmd, cap * 3.6)
        return np.clip(np.asarray(safe.act(e) if safe else idm.act(), np.float32)[:2], -1, 1)
    return act


def round_tasks(model, rnd):
    """Deterministic task list of one round: 8 episodes at each of the 5 densities. -> [(density, k, seed, weather, c_or_None)]"""
    from bev_rl.adaptive.caution import c_auto, RET_PATH
    table = json.loads(RET_PATH.read_text()); cfg = MODELS[model]; out = []
    for d in DENSITIES:
        for k in range(40):
            rng = np.random.default_rng([rnd, int(round(d * 100)), k, 7 if model == "ov" else 11])
            seed = int(rng.integers(2000, 3000)); w = cfg["weather"][int(rng.integers(len(cfg["weather"])))]
            c = c_auto(w[0], w[1], table) if cfg["uses_c"] else None          # teacher caution = true caution of the weather; the student only gets the SENSED value
            out.append((d, k, seed, w, c))
    return out


def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try: fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd); return True
    except FileExistsError: return False


FROZEN = [("bev_state/camera_grid_builder.py"), ("bev_state/assembler.py"), ("bev_state/__init__.py"), ("bev_rl/adaptive/visibility_sensor.py"), ("bev_rl/adaptive/results/visibility_sensor/weights.json"),
          ("bev_rl/adaptive/runs/ov_split/best.pt"), ("bev_rl/adaptive/runs/caut_a/best.pt"), ("weights/yolov8n.pt"), ("bev_rl/adaptive/results/retention_table.json")]


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(a=None):
    """Write the frozen-implementation manifest ONCE (never overwrites). Training refuses to run if any listed file later differs."""
    m = ROOT / "FROZEN_MANIFEST.json"; ROOT.mkdir(parents=True, exist_ok=True)
    if m.exists(): print("manifest exists, not overwritten"); return
    m.write_text(json.dumps({"files": {p: _sha(p) for p in FROZEN}}, indent=1)); print("manifest written", m)


def verify_frozen():
    m = ROOT / "FROZEN_MANIFEST.json"
    if not m.exists(): raise SystemExit("no FROZEN_MANIFEST.json: run the freeze command first")
    bad = [p for p, h in json.loads(m.read_text())["files"].items() if _sha(p) != h]
    if bad: raise SystemExit(f"frozen files changed: {bad}")


MAX_WORKERS = int(os.environ.get("CAM_MAX_WORKERS", 3))          # memory: one camera worker holds up to ~4 GB of render buffers; 4 did not fit in 13.7 GB


def collect(a):
    if a.worker > MAX_WORKERS: return
    verify_frozen()
    import torch
    torch.set_num_threads(2)
    cfg = MODELS[a.model]; rdir = ROOT / cfg["name"] / f"round{a.round}"; beta = 1.0 if a.round == 0 else 0.5 ** a.round
    net = mu = sd = None
    if a.round > 0: net, mu, sd = load_net(ROOT / cfg["name"] / f"iter_{a.round - 1}.pt", cfg["uses_c"], torch)
    env, cur_d = None, None
    for d, k, seed, w, c in round_tasks(a.model, a.round):
        name = f"d{d}_k{k}_s{seed}"
        if (rdir / f"{name}.npz").exists() or not claim(rdir / "claims" / name): continue
        if cur_d != d:
            if env is not None: env.md.close()
            env, cur_d = make_env(d, 600, a.round * 100 + a.worker, sensor=cfg["uses_c"]), d
        rng = np.random.default_rng([a.round, int(round(d * 100)), k, 99]); t0 = time.time(); cc = 0.0 if c is None else c
        o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
        if env.sensor is not None: env.sensor.reset(); env.c_hat = 0.0
        act_teacher = teacher_driver(cfg["teacher"], env, cc); B, S, Y, CH = [], [], [], []
        from bev_rl.adaptive.env import SCALE
        step = 0
        while True:
            teacher_drives = net is None or rng.random() < beta
            bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); ea = act_teacher(env)               # FULL camera pipeline EVERY step; label = clean expert action at this state
            ch = float(env.c_hat) if cfg["uses_c"] else 0.0                                                          # ch = caution the STUDENT senses from the picture
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); Y.append(ea); CH.append(ch)                  # EVERY step is recorded
            act = ea if teacher_drives else student_act(net, cfg["uses_c"], torch, bev, raw, ch, mu, sd)
            step += 1
            act = np.clip(act + rng.normal(0, a.noise, 2), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(act)
            if te or tr: break
        rdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(rdir / f"{name}.npz", B=np.stack(B), S=np.stack(S), Y=np.stack(Y), C=np.asarray(CH, np.float32), c=np.float32(cc), weather=f"{w[0]}:{w[1]}", density=d, seed=seed, outcome=str(info["outcome"]))
        print(json.dumps({"round": a.round, "density": d, "seed": seed, "weather": f"{w[0]}:{w[1]}", "c": round(cc, 2), "steps": step, "recorded": len(B), "outcome": info["outcome"], "wall_s": round(time.time() - t0)}), flush=True)
    if env is not None: env.md.close()


def train(a):
    import torch
    from bev_rl.adaptive.caution import KEEP
    cfg = MODELS[a.model]; run = ROOT / cfg["name"]
    net, mu, sd = load_net(cfg["init"] if a.round == 0 else run / f"iter_{a.round - 1}.pt", cfg["uses_c"], torch)
    B, S, C, Y = [], [], [], []
    for r in range(a.round + 1):
        for f in sorted((run / f"round{r}").glob("*.npz")):
            z = np.load(f); B.append(z["B"]); S.append(z["S"]); C.append(z["C"] if "C" in z.files else np.full(len(z["B"]), float(z["c"]), np.float32)); Y.append(z["Y"])
    B, S, C, Y = map(np.concatenate, (B, S, C, Y)); print("samples", len(B), flush=True)
    Bt = torch.from_numpy(B).float().div(10).unsqueeze(1); St = torch.from_numpy((((S - mu) / sd) * KEEP).astype(np.float32)); Ct = torch.from_numpy(C)[:, None]; Yt = torch.from_numpy(Y)
    rng = np.random.default_rng(a.round); torch.manual_seed(a.round); opt = torch.optim.Adam(net.parameters(), 1e-3); idx = np.arange(len(Yt)); net.train()
    for ep in range(cfg["epochs0"] if a.round == 0 else cfg["epochs"]):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; out = net(Bt[j], St[j], Ct[j]) if cfg["uses_c"] else net(Bt[j], St[j]); loss = ((out - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); torch.save({"model_state": net.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist(), "in_channels": 1, "weather_dim": 1 if cfg["uses_c"] else 0, "round": a.round},
                           run / f"iter_{a.round}.pt")
    (run / f"train_round{a.round}.json").write_text(json.dumps({"samples": int(len(B)), "final_loss": float(loss), "init": cfg["init"] if a.round == 0 else f"iter_{a.round - 1}.pt"}))
    print("trained round", a.round, "loss", float(loss), flush=True)


def validate(a):
    if a.worker > MAX_WORKERS: return
    verify_frozen()
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import c_auto
    from bev_rl.adaptive.env import SCALE
    cfg = MODELS[a.model]; run = ROOT / cfg["name"]; vdir = run / f"val_round{a.round}"
    net, mu, sd = load_net(run / f"iter_{a.round}.pt", cfg["uses_c"], torch); env = make_env(0.05, cfg["val_horizon"], 5000 + a.round, sensor=cfg["uses_c"])
    for w in cfg["val_weather"]:
        for seed in VAL_SEEDS:
            name = f"{w[0]}_{w[1]}__s{seed}"
            if (vdir / f"{name}.json").exists() or not claim(vdir / "claims" / name): continue
            c = c_auto(w[0], w[1], env.table) if cfg["uses_c"] else 0.0; t0 = time.time()
            o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
            if env.sensor is not None: env.sensor.reset(); env.c_hat = 0.0
            while True:
                bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); o, r, te, tr, info = env.step(student_act(net, cfg["uses_c"], torch, bev, raw, env.c_hat if cfg["uses_c"] else 0.0, mu, sd))
                if te or tr: break
            (vdir / f"{name}.json").write_text(json.dumps({"seed": seed, "weather": f"{w[0]}:{w[1]}", "outcome": info["outcome"], "steps": int(info.get("steps", 0)), "wall_s": round(time.time() - t0)}))
            print(name, info["outcome"], flush=True)
    env.md.close()


def select(a):
    import shutil
    cfg = MODELS[a.model]; run = ROOT / cfg["name"]; R = [json.loads(f.read_text()) for f in (run / f"val_round{a.round}").glob("*.json")]
    n = len(R); succ = sum(r["outcome"] == "success" for r in R) / n; fail = sum(r["outcome"] in ("crash", "off_road") for r in R) / n
    log = json.loads((run / "log.json").read_text()) if (run / "log.json").exists() else []
    log = [x for x in log if x["round"] != a.round] + [{"round": a.round, "val_n": n, "val_success": succ, "val_fail": fail}]
    best = json.loads((run / "best.json").read_text()) if (run / "best.json").exists() else None
    if best is None or (succ, -fail) > (best["val_success"], -best["val_fail"]):
        shutil.copy(run / f"iter_{a.round}.pt", run / "best.pt"); (run / "best.json").write_text(json.dumps({"round": a.round, "val_success": succ, "val_fail": fail}))
    (run / "log.json").write_text(json.dumps(sorted(log, key=lambda x: x["round"]), indent=1)); print("round", a.round, "val success", succ, "fail", fail, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "train", "validate", "select", "freeze"]); ap.add_argument("--model", default="ov", choices=list(MODELS))
    ap.add_argument("--round", type=int, default=0); ap.add_argument("--worker", type=int, default=1); ap.add_argument("--noise", type=float, default=0.1)
    a = ap.parse_args(); {"collect": collect, "train": train, "validate": validate, "select": select, "freeze": freeze}[a.cmd](a)
