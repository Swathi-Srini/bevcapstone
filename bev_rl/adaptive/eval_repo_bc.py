"""Evaluate the repo's committed BC checkpoints (and the IDM expert) in the fast env under the repo's evaluation contract.

Contract (README 'BC-to-RL handoff baseline'): clear weather, seeds 1000-1009, 800 steps, traffic density 0.05.
The BEV is built with the unmodified GitHub grid builder from ideal (simulator) vehicle detections - see env.truth_objects.

    python -m bev_rl.adaptive.eval_repo_bc --densities 0.05 0.2
"""
import argparse
import json
from pathlib import Path

import numpy as np


def run(env, act_fn, seeds):
    rows = []
    for s in seeds:
        o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
        while True:
            o, r, te, tr, i = env.step(act_fn(o))
            if te or tr:
                break
        rows.append(i)
    return rows


def summarize(name, dens, rows):
    out = [x["outcome"] for x in rows]
    d = {"policy": name, "density": dens, "n": len(rows), "success": out.count("success") / len(rows), "crash": out.count("crash") / len(rows),
         "off_road": out.count("off_road") / len(rows), "timeout": out.count("timeout") / len(rows),
         "route_completion": float(np.mean([x["route_completion"] for x in rows])), "mean_speed": float(np.mean([x["mean_speed"] for x in rows])),
         "wh_per_km": float(np.nanmean([x["energy_wh"] / (x["distance_m"] / 1000) if x["distance_m"] > 50 else np.nan for x in rows]))}
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}), flush=True)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--densities", type=float, nargs="+", default=[0.05, 0.2]); p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--out", type=Path, default=Path("bev_rl/adaptive/results/repo_baselines.json"))
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-")); seeds = list(range(lo, hi + 1))
    import torch, ultralytics  # noqa
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.env import AdaptiveEnv, SCALE
    results = []
    for dens in a.densities:
        for name in ("bc_sparse_traffic_5eps", "bc_clear_5eps_without_traffic"):
            ck = torch.load(f"bev_policy/checkpoints/{name}.pt", map_location="cpu", weights_only=True)
            net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
            mean, std = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
            env = AdaptiveEnv(seeds, [("clear", 0.0)], sequential=True, horizon=800, traffic_density=dens)

            def act(o, env=env):
                bev = torch.from_numpy(env.bev()[None, None]); sc = torch.from_numpy(((o[:6] * SCALE - mean) / std)[None].astype(np.float32))
                with torch.no_grad():
                    return net(bev, sc).squeeze(0).numpy()
            results.append(summarize(name, dens, run(env, act, seeds))); env.close()
        env = AdaptiveEnv(seeds, [("clear", 0.0)], sequential=True, horizon=800, expert=True, traffic_density=dens)
        results.append(summarize("IDM_expert", dens, run(env, lambda o: [0.0, 0.0], seeds))); env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
