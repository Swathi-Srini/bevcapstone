"""Fine-tune the proposed model on grids built by the FULL camera pipeline (dataset aggregation through cameras -> weather filter -> stereo depth + YOLO -> grid).

Why: the proposed model (ov_split) was trained on grids built from the simulator's vehicle list in clear weather. The camera pipeline gives a different grid wherever a vehicle is
present (see compare_grids.py). Same recipe as the original training (dagger_base.py --overlay --safe-teacher): the privileged collision-aware teacher (SafeTeacher) labels every step,
the student drives with probability 1-beta, action noise 0.1; the difference is that the student's INPUT is the camera grid.

    python -m bev_rl.adaptive.e2e_dagger collect --round 0 --worker 1 --density 0.05 --ckpt bev_rl/adaptive/runs/ov_split/best.pt --beta 0.5 --out bev_rl/adaptive/results/e2e_dagger
    python -m bev_rl.adaptive.e2e_dagger train   --rounds 0 1 --init bev_rl/adaptive/runs/ov_split/best.pt --name e2e_ov_split_r1 --out bev_rl/adaptive/results/e2e_dagger
Training seeds 2000-2999 (never used for evaluation, which uses 5000+). Weather per episode: clear x2, fog 1.0, rain 0.6, rain 1.0. Horizon 600 as in the original training.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]


def load_ckpt(path, torch):
    from bev_policy.model import BEVScalarPolicy
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = BEVScalarPolicy(1, 6, 0); net.load_state_dict(ck["model_state"]); net.eval()
    return net, np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)


def collect(a):
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.caution import KEEP, prep
    from bev_rl.adaptive.e2e_pipeline import E2EEnv
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.safe_teacher import SafeTeacher
    from metadrive.policy.idm_policy import IDMPolicy
    net, mu, sd = load_ckpt(a.ckpt, torch)
    rdir = a.out / f"round{a.round}"; (rdir / "claims").mkdir(parents=True, exist_ok=True)
    env = E2EEnv(list(range(2000, 3000)), [("clear", 0.0)], traffic_density=a.density, horizon=600, sample_seed=a.round * 100 + a.worker)
    rng = np.random.default_rng(1000 * a.round + a.worker)
    deadline = time.time() + a.hours * 3600
    order = np.random.default_rng(a.round).permutation(np.arange(2000, 3000))
    for k, seed in enumerate(order):
        if time.time() > deadline: break
        name = f"d{a.density}_s{seed}"
        res = rdir / f"{name}.npz"
        try:
            fd = os.open(rdir / "claims" / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        w = WEATHER[int(rng.integers(len(WEATHER)))]; t0 = time.time()
        o, _ = env.reset(options={"scenario_seed": int(seed), "weather": w}); env.drng = np.random.default_rng(int(seed) * 7 + 1)
        idm = IDMPolicy(env.md.agent, env.md.engine.global_seed); idm.enable_lane_change = False; st = SafeTeacher(idm)
        B, S, Y = [], [], []
        while True:
            bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)                    # camera grid
            ea = np.clip(np.asarray(st.act(env), np.float32)[:2], -1, 1)
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); Y.append(ea)
            if rng.random() < a.beta: act = ea
            else:
                with torch.no_grad(): act = net(*prep(torch, bev, raw, 0.0, mu, sd)).squeeze(0).numpy()
            act = np.clip(act + rng.normal(0, a.noise, 2), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(act)
            if te or tr: break
        np.savez_compressed(res, B=np.stack(B), S=np.stack(S), Y=np.stack(Y), weather=f"{w[0]}:{w[1]}", density=a.density, seed=int(seed), outcome=str(info["outcome"]))
        print(json.dumps({"seed": int(seed), "weather": f"{w[0]}:{w[1]}", "density": a.density, "steps": len(B), "outcome": info["outcome"], "wall_s": round(time.time() - t0)}), flush=True)
    env.md.close()


def train(a):
    import torch
    from bev_rl.adaptive.caution import KEEP
    net, mu, sd = load_ckpt(a.init, torch)
    B, S, Y, meta = [], [], [], []
    for r in a.rounds:
        for f in sorted((a.out / f"round{r}").glob("*.npz")):
            z = np.load(f); B.append(z["B"]); S.append(z["S"]); Y.append(z["Y"]); meta.append((str(z["weather"]), len(z["B"]), str(z["outcome"])))
    B, S, Y = np.concatenate(B), np.concatenate(S), np.concatenate(Y)
    print("samples", len(B), "episodes", len(meta), flush=True)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    opt = torch.optim.Adam(net.parameters(), a.lr)
    Bt = torch.from_numpy(B).float().div(10).unsqueeze(1); St = torch.from_numpy((((S - mu) / sd) * KEEP).astype(np.float32)); Yt = torch.from_numpy(Y)
    idx = np.arange(len(Bt)); net.train(); losses = []
    for ep in range(a.epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; loss = ((net(Bt[j], St[j]) - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss)); print("epoch", ep, "last-batch loss", round(float(loss), 4), flush=True)
    net.eval(); run = Path("bev_rl/adaptive/runs") / a.name; run.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": net.state_dict(), "scalar_mean": mu, "scalar_std": sd, "in_channels": 1, "weather_dim": 0}, run / "best.pt")
    (run / "log.json").write_text(json.dumps({"init": a.init, "rounds": a.rounds, "samples": int(len(B)), "episodes": len(meta), "epochs": a.epochs, "lr": a.lr, "losses": losses, "episodes_meta": meta}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect"); c.add_argument("--round", type=int, required=True); c.add_argument("--worker", type=int, required=True); c.add_argument("--density", type=float, required=True)
    c.add_argument("--ckpt", required=True); c.add_argument("--beta", type=float, required=True); c.add_argument("--noise", type=float, default=0.1); c.add_argument("--hours", type=float, default=3.0)
    c.add_argument("--out", type=Path, required=True)
    t = sp.add_parser("train"); t.add_argument("--rounds", type=int, nargs="+", required=True); t.add_argument("--init", required=True); t.add_argument("--name", required=True)
    t.add_argument("--epochs", type=int, default=8); t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); {"collect": collect, "train": train}[a.cmd](a)
