"""Diagnostic: what is the success ceiling of the IDM teacher in traffic, and does its lane-changing cause the crashes?
Variants (all clear weather, traffic density --density, seeds --seed0.. --n, same seeds for every variant):
  stock          : MetaDrive IDMPolicy as used by every DAgger run so far
  no_lane_change : enable_lane_change=False (pure lane keeping + car following)
  no_lc_defensive: no lane change + TIME_WANTED 2.2 s, DISTANCE_WANTED 14 m
Writes results/expert_ceiling_lane_change_ablation.json.
    python -m bev_rl.adaptive.expert_ceiling_lane_change_ablation --n 20 --density 0.2
"""
import argparse, json
from pathlib import Path
import numpy as np

VARIANTS = {"stock": {}, "no_lane_change": {"enable_lane_change": False},
            "no_lc_defensive": {"enable_lane_change": False, "TIME_WANTED": 2.2, "DISTANCE_WANTED": 14.0}}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed0", type=int, default=3100)
    ap.add_argument("--density", type=float, default=0.2); ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.env import AdaptiveEnv
    env = AdaptiveEnv(range(a.seed0, a.seed0 + a.n), [("clear", 0.0)], weather_input="zeroed", horizon=800, traffic_density=a.density)
    out = {}
    for name in a.variants:
        rows = []
        for s in range(a.seed0, a.seed0 + a.n):
            o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
            pol = IDMPolicy(env.md.agent, env.md.engine.global_seed)
            for k, v in VARIANTS[name].items(): setattr(pol, k, v)
            while True:
                act = np.clip(np.asarray(pol.act(), np.float32)[:2], -1, 1)
                o, r, te, tr, info = env.step(act)
                if te or tr: break
            rows.append({"seed": s, "outcome": info["outcome"], "speed": info["mean_speed"], "route": info["route_completion"], "wh_km": info["energy_wh"] / max(info["distance_m"] / 1000, 1e-3)})
            print(name, s, rows[-1], flush=True)
        oc = [r["outcome"] for r in rows]
        out[name] = {"n": len(rows), **{k: oc.count(k) / len(rows) for k in ("success", "crash", "off_road", "timeout")},
                     "mean_speed": float(np.mean([r["speed"] for r in rows])), "mean_wh_per_km": float(np.mean([r["wh_km"] for r in rows])), "episodes": rows}
        print("SUMMARY", name, {k: v for k, v in out[name].items() if k != "episodes"}, flush=True)
    p = Path("bev_rl/adaptive/results") / f"expert_ceiling_lane_change_ablation_d{a.density}.json"; p.write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
