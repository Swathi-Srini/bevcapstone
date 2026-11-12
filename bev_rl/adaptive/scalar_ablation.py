"""Scalar-state ablation: how does driving change when parts of the 6-D scalar state are removed?

Zero traffic => the GitHub BEV (1 channel: unknown/free/ego/detected-vehicle cells, no lane or road) is a CONSTANT grid
(verified earlier: 1 unique BEV in 4589 frames), so the BEV policy's information equals the 6-D scalar. This script therefore
trains behaviour-cloning policies (IDM expert labels) on the scalar subsets and evaluates them closed loop.

    python -m bev_rl.adaptive.scalar_ablation collect     # IDM demonstrations (clear, no traffic)
    python -m bev_rl.adaptive.scalar_ablation run         # train + evaluate every variant
"""
import csv
import sys
from pathlib import Path

import numpy as np

NAMES = ["speed", "route_progress", "lateral", "heading", "curvature", "dist_to_goal"]
VARIANTS = {"full": [], "no_speed": [0], "no_route_progress": [1], "no_lateral": [2], "no_heading": [3], "no_curvature": [4],
            "no_dist_to_goal": [5], "no_lateral_no_heading": [2, 3], "no_scalar(BEV only)": [0, 1, 2, 3, 4, 5]}
DATA = Path("bev_rl/adaptive/data/idm_clear.npz")
OUT = Path("bev_rl/adaptive/results/scalar_ablation")


def collect(n_eps=150):
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv
    e = AdaptiveEnv(range(2000, 3000), [("clear", 0.0)], expert=True, sample_seed=11, horizon=500)
    X, Y, EP = [], [], []
    for ep in range(n_eps):
        o, _ = e.reset()
        while True:
            x = o[:6].copy()
            o, r, te, tr, i = e.step([0.0, 0.0])
            X.append(x); Y.append(e.prev_action.copy()); EP.append(ep)
            if te or tr:
                break
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, X=np.array(X, np.float32), Y=np.array(Y, np.float32), EP=np.array(EP))
    print("samples", len(X), flush=True)
    e.close()


def train_bc(X, Y, drop, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    Xm = X.copy()
    mu, sd = Xm.mean(0), np.maximum(Xm.std(0), 1e-6)
    Z = (Xm - mu) / sd
    Z[:, drop] = 0.0                      # removed feature = its mean (carries no information)
    net = nn.Sequential(nn.Linear(6, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Zt, Yt = torch.tensor(Z), torch.tensor(Y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Z))
    for epoch in range(40):
        rng.shuffle(idx)
        for b in range(0, len(idx), 256):
            j = idx[b:b + 256]
            loss = ((net(Zt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        mse = float(((net(Zt) - Yt) ** 2).mean())
    return net, mu, sd, mse


def run(seeds_train=(0, 1, 2), eval_seeds=range(1000, 1010)):
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv
    d = np.load(DATA)
    X, Y = d["X"], d["Y"]
    OUT.mkdir(parents=True, exist_ok=True)
    env = AdaptiveEnv(list(eval_seeds), [("clear", 0.0)], sequential=True, horizon=800)
    rows = []
    for name, drop in VARIANTS.items():
        for ts in seeds_train:
            net, mu, sd, mse = train_bc(X, Y, drop, ts)
            for es in eval_seeds:
                o, _ = env.reset(options={"scenario_seed": es, "weather": ("clear", 0.0)})
                steer, jerk_acc, prev_a = [], [], np.zeros(2)
                while True:
                    z = (o[:6] - mu) / sd      # X was stored in env-scaled units, so no rescaling here
                    z[drop] = 0.0
                    with torch.no_grad():
                        a = net(torch.tensor(z, dtype=torch.float32)).numpy()
                    o, r, te, tr, i = env.step(a)
                    steer.append(float(np.clip(a[0], -1, 1)))
                    if te or tr:
                        break
                dist = i["distance_m"]
                rows.append({"variant": name, "train_seed": ts, "eval_seed": es, "outcome": i["outcome"], "rc": i["route_completion"],
                             "mean_abs_lat": i["mean_abs_lat"], "max_abs_lat": i["max_abs_lat"], "mean_speed": i["mean_speed"],
                             "wh_per_km": i["energy_wh"] / (dist / 1000) if dist > 50 else float("nan"),
                             "steer_rate_rms": float(np.sqrt(np.mean(np.diff(steer) ** 2))) if len(steer) > 1 else 0.0,
                             "steps": i["steps"], "train_mse": mse})
            print(name, "train_seed", ts, "done", flush=True)
    with open(OUT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    env.close()
    summarize(rows)


def env_scale():
    from bev_rl.adaptive.env import SCALE
    return SCALE            # env obs = raw/SCALE; multiply back to raw for the z-score fitted on raw data


def summarize(rows):
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[r["variant"]].append(r)
    print("%-24s %5s %6s %7s %7s %6s %7s %6s" % ("variant", "succ", "rc", "lat", "maxlat", "v", "Wh/km", "n"))
    for k, v in g.items():
        print("%-24s %5.2f %6.3f %7.2f %7.2f %6.2f %7.1f %6d" % (
            k, np.mean([x["outcome"] == "success" for x in v]), np.mean([x["rc"] for x in v]), np.mean([x["mean_abs_lat"] for x in v]),
            np.mean([x["max_abs_lat"] for x in v]), np.mean([x["mean_speed"] for x in v]), np.nanmean([x["wh_per_km"] for x in v]), len(v)))


if __name__ == "__main__":
    {"collect": collect, "run": run}[sys.argv[1]]()
