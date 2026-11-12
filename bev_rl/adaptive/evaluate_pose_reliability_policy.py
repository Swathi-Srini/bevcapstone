"""Evaluate pose-error-augmented policies (pose_reliability_dagger.py) on the SAME protocol/seeds/levels as evaluate_localisation_error_sensitivity.py
(fresh seeds 1010-1049, density 0.05, clear, persistent pose error with random sign). Conditioned arm gets c = min(1, lateral/1.5); unconditioned gets c = 0.
Writes results/localisation_error_sensitivity_<arm>_d0.05.json (same schema as the baseline arms, so the figure script picks them up).
    python -m bev_rl.adaptive.evaluate_pose_reliability_policy --arms pose_aug_conditioned pose_aug_unconditioned --n 40
"""
import argparse, json
from pathlib import Path
import numpy as np
from bev_rl.adaptive.evaluate_localisation_error_sensitivity import LEVELS, wilson
from bev_rl.adaptive.pose_reliability_dagger import LAT_MAX


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1010); ap.add_argument("--density", type=float, default=0.05)
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import spec_bev
    from bev_rl.adaptive.caution import CautionEnv, prep
    from bev_rl.adaptive.env import SCALE
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), [("clear", 0.0)], dropout=False, traffic_density=a.density, horizon=800, sample_seed=7)
    for arm in a.arms:
        ck = torch.load(f"bev_rl/adaptive/runs/{arm}/best.pt", map_location="cpu", weights_only=True); cond = bool(ck.get("condition", False))
        net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32); cells, rows = {}, []
        for dx, dth in LEVELS:
            res = []; c = min(1.0, dx / LAT_MAX) if cond else 0.0
            for s in range(a.seed0, a.seed0 + a.n):
                rng = np.random.default_rng(s * 13 + 5); sx, st = rng.choice([-1.0, 1.0], 2)       # identical error draw to the baseline evaluation
                o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)}); spec_bev.POSE_ERROR = (sx * dx, 0.0, st * dth)
                while True:
                    bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)
                    with torch.no_grad(): act = net(*prep(torch, bev, raw, c, mu, sd)).squeeze(0).numpy()
                    o, r, te, tr, i = env.step(act)
                    if te or tr: break
                spec_bev.POSE_ERROR = None
                res.append(i["outcome"]); rows.append({"arm": arm, "lat_m": dx, "heading_rad": dth, "seed": s, "outcome": i["outcome"], "route": i["route_completion"]})
            n = len(res); k = res.count("success"); key = f"lat{dx}_head{dth}"
            cells[key] = {"n": n, "lat_m": dx, "heading_rad": dth, "success": k / n, "success_ci": wilson(k, n), "crash": res.count("crash") / n, "off_road": res.count("off_road") / n, "timeout": res.count("timeout") / n, "c_fed": c}
            print(arm, key, cells[key], flush=True)
        Path("bev_rl/adaptive/results", f"localisation_error_sensitivity_{arm}_d{a.density}.json").write_text(json.dumps({"arm": arm, "cells": cells, "episodes": rows}, indent=1))

if __name__ == "__main__":
    main()
