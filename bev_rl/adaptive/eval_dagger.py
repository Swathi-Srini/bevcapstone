"""Final evaluation of a DAgger checkpoint on the RESERVED seeds (use once per selected model) with Wilson 95% intervals.

    python -m bev_rl.adaptive.eval_dagger --run base_a --densities 0.05 0.2 [--overlay] [--no-bev] [--scalar-keep 0,1,2,3,4,5]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--ckpt", default="best")
    p.add_argument("--densities", type=float, nargs="+", default=[0.05, 0.2]); p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--overlay", action="store_true"); p.add_argument("--no-bev", action="store_true")
    p.add_argument("--scalar-keep", default="0,1,2,3,4,5"); p.add_argument("--input-noise", default="0,0,0,0,0,0")
    a = p.parse_args()
    lo, hi = map(int, a.seeds.split("-")); seeds = list(range(lo, hi + 1))
    import torch, ultralytics  # noqa
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import dagger_base as D
    from bev_rl.adaptive.env import AdaptiveEnv, SCALE
    D.CFG["use_bev"] = not a.no_bev; D.CFG["overlay"] = a.overlay
    D.CFG["mask"] = np.array([1.0 if str(i) in a.scalar_keep.split(",") else 0.0 for i in range(6)], np.float32)
    D.CFG["sigma"] = np.array([float(x) for x in a.input_noise.split(",")], np.float32)
    ck = torch.load(Path("bev_rl/adaptive/runs") / a.run / f"{a.ckpt}.pt", weights_only=True)
    net = BEVScalarPolicy(1, 6, 0); net.load_state_dict(ck["model_state"]); net.eval()
    mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
    out = []
    for dens in a.densities:
        env = AdaptiveEnv(seeds, [("clear", 0.0)], sequential=True, horizon=800, traffic_density=dens); env.overlay = a.overlay
        rng = np.random.default_rng(999); res = []
        for s in seeds:
            o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
            while True:
                raw = (o[:6] * SCALE + rng.normal(0, 1, 6) * D.CFG["sigma"]).astype(np.float32)
                with torch.no_grad():
                    act = net(*D.prep(torch, env.bev(), raw, mu, sd)).squeeze(0).numpy()
                o, r, te, tr, i = env.step(act)
                if te or tr:
                    break
            res.append(i)
        env.close()
        oc = [x["outcome"] for x in res]; n = len(oc)
        row = {"run": a.run, "ckpt": a.ckpt, "density": dens, "n": n, "success": oc.count("success") / n, "success_ci95": wilson(oc.count("success"), n),
               "crash": oc.count("crash") / n, "off_road": oc.count("off_road") / n, "timeout": oc.count("timeout") / n,
               "mean_speed": float(np.mean([x["mean_speed"] for x in res])),
               "wh_per_km": float(np.nanmean([x["energy_wh"] / (x["distance_m"] / 1000) if x["distance_m"] > 50 else np.nan for x in res]))}
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        out.append(row)
    Path("bev_rl/adaptive/results").mkdir(parents=True, exist_ok=True)
    (Path("bev_rl/adaptive/results") / f"final_{a.run}_{a.ckpt}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
