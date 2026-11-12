"""Reserved-seed evaluation of weather-aware PPO policies AND of DAgger bases, under the SAME weather x density cells.

Weather acts only through the measured detector dropout (retention_table.json) on vehicles in the BEV; no dynamics change.
Seeds 1000-1009 are used once per selected model (selection used validation seeds 3100-3109).

    python -m bev_rl.adaptive.eval_ppo --policy ppo --run ppo_cond --mode conditioned
    python -m bev_rl.adaptive.eval_ppo --policy dagger --run base_a
"""
import argparse
import json
from pathlib import Path

import numpy as np

CELLS = [("clear", 0.0), ("fog", 0.8), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0)]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=("ppo", "dagger"), required=True); p.add_argument("--run", required=True)
    p.add_argument("--base", default="bev_rl/adaptive/runs/base_a/best.pt", help="checkpoint supplying scalar normalisation (ppo)")
    p.add_argument("--mode", choices=("conditioned", "unconditioned"), default="conditioned")
    p.add_argument("--ckpt", default="best"); p.add_argument("--densities", type=float, nargs="+", default=[0.05, 0.2])
    p.add_argument("--seeds", default="1000-1009"); p.add_argument("--overlay", action="store_true"); p.add_argument("--scalar-keep", default="0,1,2,3,4,5")
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-")); seeds = list(range(lo, hi + 1))
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.ppo_bev import TrafficBEVEnv, mask_of
    mask = mask_of(a)
    if a.policy == "ppo":
        from stable_baselines3 import PPO
        from bev_rl.policy import BEVActorCritic  # noqa: F401
        ck = torch.load(a.base, weights_only=True)
        model = PPO.load(Path("bev_rl/adaptive/runs") / a.run / f"{a.ckpt}.zip", device="cpu")
        wi = "zeroed" if a.mode == "unconditioned" else "explicit"
        act = lambda o: model.predict(o, deterministic=True)[0]
    else:
        from bev_policy.model import BEVScalarPolicy
        ck = torch.load(Path("bev_rl/adaptive/runs") / a.run / f"{a.ckpt}.pt", weights_only=True)
        net = BEVScalarPolicy(1, 6, 0); net.load_state_dict(ck["model_state"]); net.eval(); wi = "zeroed"

        def act(o):
            with torch.no_grad():
                return net(torch.from_numpy(o["bev"][None]), torch.from_numpy(o["scalar"][None])).squeeze(0).numpy()
    rows = []
    for dens in a.densities:
        for w in CELLS:
            env = TrafficBEVEnv(seeds, [w], ck["scalar_mean"], ck["scalar_std"], wi, dens, True, overlay=a.overlay, mask=mask,
                                sequential=True, horizon=800, sample_seed=7)
            res = []
            for s in seeds:
                o, _ = env.reset(options={"scenario_seed": s, "weather": w})
                while True:
                    o, r, te, tr, i = env.step(act(o))
                    if te or tr:
                        break
                res.append(i)
            env.close()
            oc = [x["outcome"] for x in res]; n = len(oc)
            row = {"policy": a.run, "density": dens, "weather": f"{w[0]}:{w[1]}", "n": n, "success": oc.count("success") / n,
                   "success_ci95": wilson(oc.count("success"), n), "crash": oc.count("crash") / n, "off_road": oc.count("off_road") / n,
                   "timeout": oc.count("timeout") / n, "mean_speed": float(np.mean([x["mean_speed"] for x in res])),
                   "wh_per_km": float(np.nanmean([x["energy_wh"] / (x["distance_m"] / 1000) if x["distance_m"] > 50 else np.nan for x in res]))}
            rows.append(row); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    Path("bev_rl/adaptive/results").mkdir(parents=True, exist_ok=True)
    (Path("bev_rl/adaptive/results") / f"final_weather_{a.run}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
