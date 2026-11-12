"""DAgger training of the caution-conditioned policy (see caution.py).

    python -m bev_rl.adaptive.caution_dagger --name caut_a --iters 5

Per episode: weather ~ TRAIN_WEATHER (+clear), density ~ DENSITIES, caution c ~ 50% U(0,1) / 50% c_auto(weather).
Model selection: validation seeds 3100-3109 only (reserved/eval seeds are never touched here).
Selection score = mean over {clear, rain 0.6, rain 1.0} @ density 0.05 (sparse-traffic focus since 2026-11-05) of success under c_auto (ties: lower failure).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bev_rl.adaptive.caution import CautionEnv, KEEP, c_auto, episode
from bev_rl.adaptive.env import SCALE

DENSITIES = [0.0, 0.05, 0.1, 0.2, 0.3]
TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.6), ("fog", 1.0), ("rain", 0.2), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("rain", 0.6), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3110))


def train(net, B, S, C, Y, mu, sd, torch, epochs, lr, rng):
    opt = torch.optim.Adam(net.parameters(), lr)
    Bt = torch.from_numpy(np.stack(B)).float().div(10).unsqueeze(1)
    St = torch.from_numpy((((np.stack(S) - mu) / sd) * KEEP).astype(np.float32))
    Ct = torch.tensor(np.array(C, np.float32))[:, None]; Yt = torch.from_numpy(np.stack(Y))
    idx = np.arange(len(Bt)); net.train()
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            loss = ((net(Bt[j], St[j], Ct[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return float(loss)


def validate(net, mu, sd, torch, table_env_kwargs, n, no_caution=False):
    env = CautionEnv(VAL_SEEDS, VAL_WEATHER, traffic_density=0.05, horizon=600, **table_env_kwargs)
    res = []
    for w in VAL_WEATHER:
        for s in VAL_SEEDS[:n]:
            *_, info = episode(env, net, mu, sd, torch, s, w, 0.0 if no_caution else c_auto(*w, env.table))
            res.append((info["outcome"] == "success", info["outcome"] in ("crash", "off_road")))
    env.close()
    a = np.array(res, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--iters", type=int, default=5)
    p.add_argument("--eps-per-iter", type=int, default=40); p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-n", type=int, default=10)
    p.add_argument("--no-caution", action="store_true", help="control: c is always fed as 0 and the teacher is stock IDM (dropout still applied)")
    a = p.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    net = BEVScalarPolicy(1, 6, 1)           # weather branch reused as the 1-D caution input
    B, S, C, Y = [], [], [], []
    mu = sd = None
    best, log, t0 = (-1.0, 1.0), [], time.time()
    for it in range(a.iters + 1):
        beta = 1.0 if it == 0 else 0.5 ** it
        stats = []
        for d in DENSITIES:
            env = CautionEnv(range(2000, 3000), TRAIN_WEATHER, traffic_density=d, horizon=600, sample_seed=a.seed * 10 + it)
            for k in range(max(1, a.eps_per_iter // len(DENSITIES))):
                w = TRAIN_WEATHER[int(rng.integers(len(TRAIN_WEATHER)))]
                c = 0.0 if a.no_caution else (float(rng.random()) if rng.random() < 0.5 else c_auto(*w, env.table))
                b, s, cc, y, info = episode(env, None if it == 0 else net, mu, sd, torch, int(rng.choice(range(2000, 3000))), w, c,
                                            beta=beta, noise=a.noise, rng=rng, collect=True)
                B += b; S += s; C += cc; Y += y; stats.append(info["outcome"])
            env.close()
        if mu is None:
            Sa = np.stack(S); mu, sd = Sa.mean(0).astype(np.float32), np.maximum(Sa.std(0), 1e-6).astype(np.float32)
        loss = train(net, B, S, C, Y, mu, sd, torch, a.epochs if it else 3 * a.epochs, 1e-3, rng)
        v = validate(net, mu, sd, torch, {}, a.val_n, a.no_caution)
        rec = {"iter": it, "beta": beta, "samples": len(B), "train_loss": loss, "rollout_outcomes": {o: stats.count(o) for o in set(stats)},
               "val_success_auto": v[0], "val_fail_auto": v[1], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        torch.save({"model_state": net.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist(), "iter": it, "val": v}, out / f"iter_{it}.pt")
        if (v[0], -v[1]) > (best[0], -best[1]):
            best = v; torch.save(torch.load(out / f"iter_{it}.pt", weights_only=True), out / "best.pt"); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
