"""Evaluate ANY caution-network checkpoint (runs/<run>/iter_<k>.pt or best.pt) on the fresh sparse-traffic protocol (seeds 1010-1049, density 0.05, clear, detector dropout at
measured rate for the chosen weather, caution input fixed to --c). Used to test whether later DAgger rounds beat round 0 (model-selection check).
    python -m bev_rl.adaptive.evaluate_checkpoint_sparse --run ctrl_nocaut --iters 0 2 5 --n 40
Writes results/checkpoint_eval_<run>_iter<k>.json
"""
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--iters", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--seed0", type=int, default=1010); ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--c", type=float, default=0.0); ap.add_argument("--weather", default="clear:0.0")
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.caution import CautionEnv, episode
    from bev_rl.adaptive.eval_caution import wilson
    kind, lvl = a.weather.split(":"); w = (kind, float(lvl))
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), [w], dropout=(kind != "clear"), traffic_density=a.density, horizon=800, sample_seed=7)
    for k in a.iters:
        ck = torch.load(f"bev_rl/adaptive/runs/{a.run}/iter_{k}.pt", map_location="cpu", weights_only=True)
        net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32); rows = []
        for s in range(a.seed0, a.seed0 + a.n):
            *_, i = episode(env, net, mu, sd, torch, s, w, a.c)
            rows.append({"seed": s, "outcome": i["outcome"], "wh_per_km": i["energy_wh"] / max(i["distance_m"] / 1000, 1e-3), "speed": i["mean_speed"]})
        oc = [r["outcome"] for r in rows]; n = len(oc); kk = oc.count("success")
        out = {"run": a.run, "iter": k, "weather": a.weather, "c": a.c, "n": n, "success": kk / n, "success_ci": wilson(kk, n), "crash": oc.count("crash") / n, "off_road": oc.count("off_road") / n,
               "timeout": oc.count("timeout") / n, "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])), "episodes": rows}
        Path("bev_rl/adaptive/results", f"checkpoint_eval_{a.run}_iter{k}.json").write_text(json.dumps(out, indent=1))
        print(a.run, "iter", k, {x: (round(out[x], 3) if isinstance(out[x], float) else out[x]) for x in ("success", "crash", "off_road", "timeout", "wh_per_km")}, flush=True)

if __name__ == "__main__":
    main()
