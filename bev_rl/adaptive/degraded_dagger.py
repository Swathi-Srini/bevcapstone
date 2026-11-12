"""DAgger of a perception-degradation-aware base policy: the student sees the CALIBRATED degraded BEV (degraded_env.py), the expert is the SafeTeacher on GROUND TRUTH.

    python -m bev_rl.adaptive.degraded_dagger --name deg_base --iters 5            # non-adaptive: BEVScalarPolicy(1,6,0), the control and the init of both PPO variants
    python -m bev_rl.adaptive.degraded_dagger --name deg_ada  --iters 5 --adaptive  # adds the weather branch Linear(4->16): weather_vector(kind, intensity) as extra input

Same input design as ov_split (overlay BEV + scalars speed/progress/goal-distance), same optimiser/loss as every earlier DAgger run (MSE, Adam 1e-3, batch 128, action noise 0.1).
Seeds: train scenarios 2000-2999, validation 3100-3139 (never used for the final evaluation, which uses 5000-5199).
Model selection: validation success averaged over {clear, fog 1.0, rain 0.6, rain 1.0}; ties -> the LATER round (the final DAgger iterate is the one with theory behind it; the earlier
'ties keep round 0' rule selected the untrained round-0 checkpoint twice).
Memory: BEVs are kept as int8 (x10) and cast per minibatch (the earlier code cast the whole dataset to float32, ~2 GB at 120k samples).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bev_rl.adaptive.caution import KEEP
from bev_rl.adaptive.degraded_env import DegradedEnv, run_episode

DENSITIES = [0.05, 0.05, 0.1, 0.0]
TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.8), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3140))


def train(net, B, S, W, Y, mu, sd, torch, epochs, lr, rng, adaptive):
    opt = torch.optim.Adam(net.parameters(), lr)
    Bt = torch.from_numpy(np.stack(B)); St = torch.from_numpy((((np.stack(S) - mu) / sd) * KEEP).astype(np.float32))
    Wt = torch.from_numpy(np.stack(W).astype(np.float32)); Yt = torch.from_numpy(np.stack(Y).astype(np.float32))
    idx = np.arange(len(Bt)); net.train(); loss = None
    for _ in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]; bb = Bt[j].float().div(10).unsqueeze(1)
            out = net(bb, St[j], Wt[j]) if adaptive else net(bb, St[j])
            loss = ((out - Yt[j]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return float(loss)


def validate(net, mu, sd, torch, adaptive, n):
    env = DegradedEnv(VAL_SEEDS, VAL_WEATHER, mode="calibrated", traffic_density=0.05, horizon=600)
    res = []
    for w in VAL_WEATHER:
        for s in VAL_SEEDS[:n]:
            *_, info = run_episode(env, net, mu, sd, torch, s, w, adaptive)
            res.append((info["outcome"] == "success", info["outcome"] in ("crash", "off_road")))
    env.close(); a = np.array(res, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--iters", type=int, default=5); p.add_argument("--eps-per-iter", type=int, default=40)
    p.add_argument("--epochs", type=int, default=6); p.add_argument("--noise", type=float, default=0.1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-n", type=int, default=15); p.add_argument("--adaptive", action="store_true")
    a = p.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    wd = 4 if a.adaptive else 0; net = BEVScalarPolicy(1, 6, wd)
    B, S, W, Y = [], [], [], []; mu = sd = None; best, log, t0 = (-1.0, 1.0, -1), [], time.time()
    for it in range(a.iters + 1):
        beta = 1.0 if it == 0 else 0.5 ** it; stats = []
        for d in DENSITIES:
            env = DegradedEnv(range(2000, 3000), TRAIN_WEATHER, mode="calibrated", traffic_density=d, horizon=600, sample_seed=a.seed * 10 + it)
            for _ in range(max(1, a.eps_per_iter // len(DENSITIES))):
                w = TRAIN_WEATHER[int(rng.integers(len(TRAIN_WEATHER)))]
                b, s, ww, y, info = run_episode(env, None if it == 0 else net, mu, sd, torch, int(rng.choice(range(2000, 3000))), w, a.adaptive,
                                                beta=beta, noise=a.noise, rng=rng, collect=True)
                B += b; S += s; W += ww; Y += y; stats.append(info["outcome"])
            env.close()
        if mu is None:
            Sa = np.stack(S); mu, sd = Sa.mean(0).astype(np.float32), np.maximum(Sa.std(0), 1e-6).astype(np.float32)
        loss = train(net, B, S, W, Y, mu, sd, torch, a.epochs if it else 3 * a.epochs, 1e-3, rng, a.adaptive)
        v = validate(net, mu, sd, torch, a.adaptive, a.val_n)
        rec = {"iter": it, "beta": beta, "samples": len(B), "train_loss": loss, "rollout_outcomes": {o: stats.count(o) for o in set(stats)},
               "val_success": v[0], "val_fail": v[1], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        ck = {"model_state": net.state_dict(), "scalar_mean": mu.tolist(), "scalar_std": sd.tolist(), "iter": it, "val": v, "in_channels": 1, "weather_dim": wd}
        torch.save(ck, out / f"iter_{it}.pt")
        if (v[0], -v[1], it) > (best[0], -best[1], best[2]):                 # ties -> later round
            best = (v[0], v[1], it); torch.save(ck, out / "best.pt"); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
