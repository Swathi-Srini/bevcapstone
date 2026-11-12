"""Does a localisation (ego-pose-vs-HD-map) error break the policies in sparse traffic? Clear weather, density 0.05, fresh seeds, no detector dropout.
Per episode a PERSISTENT pose bias is drawn (sign random, magnitude = level): lateral dx [m] and heading dtheta [rad] (levels below).
  map-overlay policy (ov_split): the overlay is drawn from the erroneous pose (spec_bev.POSE_ERROR); its scalars (speed, progress, goal) are unaffected.
  scalar policies (base_a, repo_bc_sparse): the same error is added to the lateral-offset and heading-error scalars (inputs 2 and 3).
    python -m bev_rl.adaptive.evaluate_localisation_error_sensitivity --n 40
"""
import argparse, json
from pathlib import Path
import numpy as np

LEVELS = [(0.0, 0.0), (0.25, 0.02), (0.5, 0.05), (1.0, 0.10), (2.0, 0.20)]
ARMS = {"ov_split": ("bev_rl/adaptive/runs/ov_split/best.pt", True, [0, 1, 5]), "base_a": ("bev_rl/adaptive/runs/base_a/best.pt", False, None),
        "repo_bc_sparse": ("bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", False, None)}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed0", type=int, default=1010)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import spec_bev
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.env import SCALE
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), [("clear", 0.0)], dropout=False, traffic_density=a.density, horizon=800, sample_seed=7)
    for arm in a.arms:
        path, overlay, keep = ARMS[arm]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
        mask = np.ones(6, np.float32) if keep is None else np.array([1.0 if i in keep else 0.0 for i in range(6)], np.float32)
        cells, rows = {}, []
        for dx, dth in LEVELS:
            res = []
            for s in range(a.seed0, a.seed0 + a.n):
                rng = np.random.default_rng(s * 13 + 5); sx, st = rng.choice([-1.0, 1.0], 2); ex, et = sx * dx, st * dth
                o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
                spec_bev.POSE_ERROR = (ex, 0.0, et) if overlay else None
                while True:
                    bev = env.asm.build_bev_grid(env.truth_objects())
                    if overlay: bev = spec_bev.overlay_github_bev(bev, env.md)
                    raw = (o[:6] * SCALE).astype(np.float32)
                    if not overlay: raw[2] += ex; raw[3] += et
                    sc = (((raw - mu) / sd) * mask)[None].astype(np.float32)
                    with torch.no_grad():
                        act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
                    o, r, te, tr, i = env.step(act)
                    if te or tr: break
                spec_bev.POSE_ERROR = None
                res.append(i["outcome"]); rows.append({"arm": arm, "lat_m": dx, "heading_rad": dth, "seed": s, "outcome": i["outcome"], "route": i["route_completion"]})
            n = len(res); k = res.count("success"); key = f"lat{dx}_head{dth}"
            cells[key] = {"n": n, "lat_m": dx, "heading_rad": dth, "success": k / n, "success_ci": wilson(k, n), "crash": res.count("crash") / n, "off_road": res.count("off_road") / n, "timeout": res.count("timeout") / n}
            print(arm, key, cells[key], flush=True)
        Path("bev_rl/adaptive/results", f"localisation_error_sensitivity_{arm}_d{a.density}.json").write_text(json.dumps({"arm": arm, "cells": cells, "episodes": rows}, indent=1))

if __name__ == "__main__":
    main()
