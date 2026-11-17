"""Fine-tune BOTH existing policies (proposed ov_split and caution-aware caut_a) on grids from the full camera pipeline + CameraGridBuilder.

Dataset aggregation, same recipe as the original trainings, only the student's INPUT changes (camera grid instead of the simulator-list grid):
    * every step the privileged teachers label the visited state: SafeTeacher (collision-aware IDM) for ov_split, caution IDM (30(1-c/3) km/h, gap 10+15c m) for caut_a;
    * one rollout serves both models: episodes alternate between the ov_split student and the caut_a student driving (beta = prob. of executing that model's teacher action; noise 0.1);
    * per episode: weather in {clear, clear, fog 1.0, rain 0.6, rain 1.0}; caution c = U(0,1) half the time, c_auto(weather) otherwise (as in the original caution training);
    * training seeds 2000-2999 only. No YOLO, depth, camera or weather code is touched; the ORIGINAL checkpoints are never overwritten (new names: e2e_ov_split_*, e2e_caut_a_*).

    python -m bev_rl.adaptive.e2e_dagger_both collect --round 0 --worker 1 --density 0.05 --beta 0.5 --out bev_rl/adaptive/results/e2e_dagger_both
    python -m bev_rl.adaptive.e2e_dagger_both train --rounds 0 --out ... --suffix r0 --init-ov bev_rl/adaptive/runs/ov_split/best.pt --init-caut bev_rl/adaptive/runs/caut_a/best.pt
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]


def make_env(density, seed_tag):
    from bev_rl.adaptive.e2e_pipeline import E2EEnv
    from bev_state.camera_grid_builder import CameraGridBuilder
    import bev_state.assembler as A

    class BuilderEnv(E2EEnv):
        def student_bev(self):
            from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
            from weather.weather_utils import apply_weather
            raw = {n: frame_from_sensor(self.md, n) for n in CAMERA_RIGS}
            kind, level = self.weather
            if kind == "clear" or level <= 0: frames = raw
            else:
                np.random.seed(20261114 + int(self.md.engine.global_seed) * 100003 + self.step_count)
                frames = {n: apply_weather(f, kind, level) for n, f in raw.items()}
                frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw["front_left_camera"], raw["front_right_camera"], kind, level)
            self.step_count += 1
            depth = A.front_stereo_depth(frames["front_left_camera"], frames["front_right_camera"], self.asm.matcher)
            dets = {n: A.run_yolo(self.yolo, frames[n]) for n in A.YOLO_CAMERAS}
            return self.builder.build(self.asm.objects_from_detections(dets, depth), self.md)
    env = BuilderEnv(list(range(2000, 3000)), [("clear", 0.0)], traffic_density=density, horizon=600, sample_seed=seed_tag)
    env.builder = CameraGridBuilder(env.asm)
    return env


def load_ov(path, torch):
    from bev_policy.model import BEVScalarPolicy
    ck = torch.load(path, map_location="cpu", weights_only=True); net = BEVScalarPolicy(1, 6, 0); net.load_state_dict(ck["model_state"]); net.eval()
    return net, np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def load_caut(path, torch):
    from bev_policy.model import BEVScalarPolicy
    ck = torch.load(path, map_location="cpu", weights_only=True); net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
    return net, np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def collect(a):
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import c_auto, prep
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.safe_teacher import SafeTeacher
    from metadrive.policy.idm_policy import IDMPolicy
    ov, mu_o, sd_o = load_ov(a.ckpt_ov, torch); ca, mu_c, sd_c = load_caut(a.ckpt_caut, torch)
    rdir = a.out / f"round{a.round}"; (rdir / "claims").mkdir(parents=True, exist_ok=True)
    env = make_env(a.density, a.round * 100 + a.worker)
    rng = np.random.default_rng(1000 * a.round + a.worker); deadline = time.time() + a.hours * 3600
    for seed in np.random.default_rng(a.round).permutation(np.arange(2000, 3000)):
        if time.time() > deadline: break
        name = f"d{a.density}_s{seed}"
        try: fd = os.open(rdir / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError: continue
        w = WEATHER[int(rng.integers(len(WEATHER)))]; t0 = time.time()
        c = float(rng.uniform(0, 1)) if rng.random() < 0.5 else c_auto(w[0], w[1], env.table)
        o, _ = env.reset(options={"scenario_seed": int(seed), "weather": w}); env.drng = np.random.default_rng(int(seed) * 7 + 1)
        idm = IDMPolicy(env.md.agent, env.md.engine.global_seed); idm.enable_lane_change = False; safe = SafeTeacher(idm); cau = env.make_expert(c)
        driver = "ov" if int(seed) % 2 == 0 else "caut"
        B, S, C, YO, YC = [], [], [], [], []
        while True:
            bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)
            yo = np.clip(np.asarray(safe.act(env), np.float32)[:2], -1, 1); yc = np.clip(np.asarray(cau.act(), np.float32)[:2], -1, 1)
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); C.append(c); YO.append(yo); YC.append(yc)
            if rng.random() < a.beta: act = yo if driver == "ov" else yc
            else:
                with torch.no_grad():
                    act = (ov(*prep(torch, bev, raw, c, mu_o, sd_o)[:2]) if driver == "ov" else ca(*prep(torch, bev, raw, c, mu_c, sd_c))).squeeze(0).numpy()
            act = np.clip(act + rng.normal(0, a.noise, 2), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(act)
            if te or tr: break
        np.savez_compressed(rdir / f"{name}.npz", B=np.stack(B), S=np.stack(S), C=np.array(C, np.float32), YO=np.stack(YO), YC=np.stack(YC),
                            weather=f"{w[0]}:{w[1]}", density=a.density, seed=int(seed), driver=driver, outcome=str(info["outcome"]))
        print(json.dumps({"seed": int(seed), "driver": driver, "weather": f"{w[0]}:{w[1]}", "c": round(c, 2), "density": a.density, "steps": len(B), "outcome": info["outcome"], "wall_s": round(time.time() - t0)}), flush=True)
    env.md.close()


def fit(net, inputs, Y, epochs, lr, seed=0):
    import torch
    rng = np.random.default_rng(seed); torch.manual_seed(seed); opt = torch.optim.Adam(net.parameters(), lr); idx = np.arange(len(Y)); net.train(); losses = []
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; loss = ((net(*[t[j] for t in inputs]) - Y[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss)); print("  epoch", ep, "last-batch loss", round(float(loss), 4), flush=True)
    net.eval(); return losses


def train(a):
    import torch
    from bev_rl.adaptive.caution import KEEP
    B, S, C, YO, YC, meta = [], [], [], [], [], []
    for r in a.rounds:
        for f in sorted((a.out / f"round{r}").glob("*.npz")):
            z = np.load(f); B.append(z["B"]); S.append(z["S"]); C.append(z["C"]); YO.append(z["YO"]); YC.append(z["YC"]); meta.append((str(z["weather"]), len(z["B"]), str(z["outcome"]), str(z["driver"])))
    B, S, C, YO, YC = map(np.concatenate, (B, S, C, YO, YC)); print("samples", len(B), "episodes", len(meta), flush=True)
    Bt = torch.from_numpy(B).float().div(10).unsqueeze(1); Ct = torch.from_numpy(C)[:, None]
    for tag, init, loader, Y, uses_c in (("ov_split", a.init_ov, load_ov, YO, False), ("caut_a", a.init_caut, load_caut, YC, True)):
        net, mu, sd = loader(init, torch); St = torch.from_numpy((((S - mu) / sd) * KEEP).astype(np.float32)); Yt = torch.from_numpy(Y)
        print("fine-tuning", tag, flush=True); losses = fit(net, (Bt, St, Ct) if uses_c else (Bt, St), Yt, a.epochs, a.lr)
        run = Path("bev_rl/adaptive/runs") / f"e2e_{tag}_{a.suffix}"; run.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": net.state_dict(), "scalar_mean": mu, "scalar_std": sd, "in_channels": 1, "weather_dim": 1 if uses_c else 0}, run / "best.pt")
        (run / "log.json").write_text(json.dumps({"init": init, "rounds": a.rounds, "samples": int(len(B)), "episodes": len(meta), "epochs": a.epochs, "lr": a.lr, "losses": losses, "episodes_meta": meta}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect"); c.add_argument("--round", type=int, required=True); c.add_argument("--worker", type=int, required=True); c.add_argument("--density", type=float, required=True)
    c.add_argument("--beta", type=float, required=True); c.add_argument("--noise", type=float, default=0.1); c.add_argument("--hours", type=float, default=3.0); c.add_argument("--out", type=Path, required=True)
    c.add_argument("--ckpt-ov", default="bev_rl/adaptive/runs/ov_split/best.pt"); c.add_argument("--ckpt-caut", default="bev_rl/adaptive/runs/caut_a/best.pt")
    t = sp.add_parser("train"); t.add_argument("--rounds", type=int, nargs="+", required=True); t.add_argument("--suffix", required=True); t.add_argument("--out", type=Path, required=True)
    t.add_argument("--init-ov", default="bev_rl/adaptive/runs/ov_split/best.pt"); t.add_argument("--init-caut", default="bev_rl/adaptive/runs/caut_a/best.pt")
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(); {"collect": collect, "train": train}[a.cmd](a)
