"""Spec-BEV ablation: with the spec's route/boundary overlay in the BEV, what does the 6-D scalar add?

    python -m bev_rl.adaptive.bev_ablation collect
    python -m bev_rl.adaptive.bev_ablation run

Policy = repo CNN+MLP (bev_policy.model.BEVScalarPolicy) trained by behaviour cloning on IDM labels (zero traffic, clear).
Variants differ only in which inputs are visible: BEV and/or scalar components (hidden input is replaced by a constant).
"""
import csv
import sys
from pathlib import Path

import numpy as np

SC = {"speed": 0, "route_progress": 1, "lateral": 2, "heading": 3, "curvature": 4, "dist_goal": 5}
VARIANTS = {                      # name: (use_bev, visible scalar indices)
    "bev+scalar(full)": (True, [0, 1, 2, 3, 4, 5]),
    "bev_only": (True, []),
    "scalar_only(const BEV)": (False, [0, 1, 2, 3, 4, 5]),
    "bev+speed": (True, [0]),
    "bev+speed+progress+goal": (True, [0, 1, 5]),
}
DATA = Path("bev_rl/adaptive/data/idm_specbev.npz")
OUT = Path("bev_rl/adaptive/results/bev_ablation")


def collect(n_eps=100):
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv, SCALE
    from bev_rl.adaptive.spec_bev import build_spec_bev
    e = AdaptiveEnv(range(2000, 3000), [("clear", 0.0)], expert=True, sample_seed=21, horizon=500)
    B, S, Y = [], [], []
    for ep in range(n_eps):
        o, _ = e.reset()
        while True:
            B.append(np.rint(build_spec_bev(e.md) * 10).astype(np.uint8)); S.append(o[:6] * SCALE)
            o, r, te, tr, i = e.step([0.0, 0.0])
            Y.append(e.prev_action.copy())
            if te or tr:
                break
        print("ep", ep, len(Y), flush=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, B=np.stack(B), S=np.array(S, np.float32), Y=np.array(Y, np.float32))
    e.close()


def make_inputs(B, S, use_bev, vis, mu, sd):
    import torch
    bev = torch.tensor(B, dtype=torch.float32) / 10.0
    if not use_bev:
        bev = torch.zeros_like(bev)
    z = (S - mu) / sd
    mask = np.zeros(6, np.float32); mask[vis] = 1.0
    return bev.unsqueeze(1), torch.tensor(z * mask, dtype=torch.float32)


def train(d, use_bev, vis, seed, epochs=12):
    import torch
    from bev_policy.model import BEVScalarPolicy
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    B, S, Y = d["B"], d["S"], torch.tensor(d["Y"])
    mu, sd = S.mean(0), np.maximum(S.std(0), 1e-6)
    net = BEVScalarPolicy(1, 6, 0)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    idx = np.arange(len(B))
    for ep in range(epochs):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            j = idx[b:b + 128]
            bev, sc = make_inputs(B[j], S[j], use_bev, vis, mu, sd)
            loss = ((net(bev, sc) - Y[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net, mu, sd, float(loss)


def run(train_seeds=(0, 1, 2), eval_seeds=range(1000, 1010)):
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.env import AdaptiveEnv, SCALE
    from bev_rl.adaptive.spec_bev import build_spec_bev
    d = dict(np.load(DATA))
    OUT.mkdir(parents=True, exist_ok=True)
    env = AdaptiveEnv(list(eval_seeds), [("clear", 0.0)], sequential=True, horizon=800)
    rows = []
    for name, (use_bev, vis) in VARIANTS.items():
        for ts in train_seeds:
            net, mu, sd, tl = train(d, use_bev, vis, ts)
            net.eval()
            for es in eval_seeds:
                o, _ = env.reset(options={"scenario_seed": es, "weather": ("clear", 0.0)})
                steer = []
                while True:
                    g = np.rint(build_spec_bev(env.md) * 10).astype(np.uint8)[None]
                    bev, sc = make_inputs(g, (o[:6] * SCALE)[None], use_bev, vis, mu, sd)
                    with torch.no_grad():
                        a = net(bev, sc).squeeze(0).numpy()
                    o, r, te, tr, i = env.step(a); steer.append(float(a[0]))
                    if te or tr:
                        break
                dist = i["distance_m"]
                rows.append({"variant": name, "train_seed": ts, "eval_seed": es, "outcome": i["outcome"], "rc": i["route_completion"],
                             "mean_abs_lat": i["mean_abs_lat"], "max_abs_lat": i["max_abs_lat"], "mean_speed": i["mean_speed"],
                             "wh_per_km": i["energy_wh"] / (dist / 1000) if dist > 50 else float("nan"),
                             "steer_rate_rms": float(np.sqrt(np.mean(np.diff(steer) ** 2))) if len(steer) > 1 else 0.0,
                             "steps": i["steps"], "train_loss": tl})
            print(name, "train_seed", ts, "done", flush=True)
    with open(OUT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    env.close()
    from bev_rl.adaptive.scalar_ablation import summarize
    summarize(rows)


if __name__ == "__main__":
    {"collect": collect, "run": run}[sys.argv[1]]()
