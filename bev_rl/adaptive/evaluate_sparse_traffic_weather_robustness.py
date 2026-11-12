"""Sparse-traffic (density 0.05) weather-robustness comparison on FRESH seeds, same seeds / weather / detector-dropout stream for every arm.

Weather acts only through the MEASURED detector retention (results/retention_table.json): each simulator vehicle is dropped from the
BEV with prob. 1 - keep_prob(weather), exactly as in CautionEnv. No weather physics. Arms:
  repo_bc_sparse  : repo bc_sparse_traffic_5eps  (GitHub BEV, 6 scalars)
  repo_bc_clear   : repo bc_clear_5eps_without_traffic
  base_a          : my DAgger, GitHub BEV + 6 scalars
  ov_split        : my DAgger, overlay BEV + 3 scalars
Reports per weather cell: success, at-fault-ish outcomes (crash/off_road), timeout, Wilson 95% CI, and RETENTION = success(weather)/success(clear).
    python -m bev_rl.adaptive.evaluate_sparse_traffic_weather_robustness --n 40
"""
import argparse, json
from pathlib import Path
import numpy as np

WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
ARMS = {"repo_bc_sparse": ("bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", False, None),
        "repo_bc_clear": ("bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt", False, None),
        "base_a": ("bev_rl/adaptive/runs/base_a/best.pt", False, None),
        "ov_split": ("bev_rl/adaptive/runs/ov_split/best.pt", True, [0, 1, 5])}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed0", type=int, default=1010)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.caution import CautionEnv, keep_prob
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.spec_bev import overlay_github_bev
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), WEATHER, dropout=True, traffic_density=a.density, horizon=800, sample_seed=7)
    out = Path("bev_rl/adaptive/results"); out.mkdir(exist_ok=True)
    for arm in a.arms:
        path, overlay, keep = ARMS[arm]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
        mask = np.ones(6, np.float32) if keep is None else np.array([1.0 if i in keep else 0.0 for i in range(6)], np.float32)
        cells, rows = {}, []
        for w in WEATHER:
            p = keep_prob(*w, env.table); res = []
            for s in range(a.seed0, a.seed0 + a.n):
                o, _ = env.reset(options={"scenario_seed": s, "weather": w}); env.drng = np.random.default_rng(s * 7 + 1)
                while True:
                    objs = [x for x in env.truth_objects() if env.drng.random() < p]
                    bev = env.asm.build_bev_grid(objs)
                    if overlay: bev = overlay_github_bev(bev, env.md)
                    sc = (((o[:6] * SCALE - mu) / sd) * mask)[None].astype(np.float32)
                    with torch.no_grad():
                        act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
                    o, r, te, tr, i = env.step(act)
                    if te or tr: break
                res.append(i["outcome"]); rows.append({"arm": arm, "weather": f"{w[0]}:{w[1]}", "seed": s, "outcome": i["outcome"], "route": i["route_completion"], "speed": i["mean_speed"]})
            n = len(res); k = res.count("success")
            cells[f"{w[0]}:{w[1]}"] = {"n": n, "keep_prob": p, "success": k / n, "success_ci": wilson(k, n), "crash_or_offroad": (res.count("crash") + res.count("off_road")) / n, "timeout": res.count("timeout") / n}
            print(arm, w, cells[f"{w[0]}:{w[1]}"], flush=True)
        base = cells["clear:0.0"]["success"]
        for c in cells.values(): c["retention_vs_clear"] = (c["success"] / base) if base > 0 else None
        (out / f"sparse_traffic_weather_robustness_{arm}_d{a.density}.json").write_text(json.dumps({"arm": arm, "cells": cells, "episodes": rows}, indent=1))

if __name__ == "__main__":
    main()
