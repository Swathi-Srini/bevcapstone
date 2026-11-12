"""Teacher ceiling: stock IDM vs IDM+predictive safety layer, clear weather, traffic density --density, same seeds.
    python -m bev_rl.adaptive.evaluate_safe_teacher_ceiling --n 20 --density 0.2
"""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed0", type=int, default=3100)
    ap.add_argument("--density", type=float, default=0.2); ap.add_argument("--t-soft", type=float, default=2.4); ap.add_argument("--t-hard", type=float, default=1.3)
    a = ap.parse_args()
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.env import AdaptiveEnv
    from bev_rl.adaptive.safe_teacher import SafeTeacher
    env = AdaptiveEnv(range(a.seed0, a.seed0 + a.n), [("clear", 0.0)], weather_input="zeroed", horizon=800, traffic_density=a.density)
    out = {}
    for name in ("safe_teacher",):
        rows = []
        for s in range(a.seed0, a.seed0 + a.n):
            env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
            idm = IDMPolicy(env.md.agent, env.md.engine.global_seed); idm.enable_lane_change = False
            pol = SafeTeacher(idm, t_soft_s=a.t_soft, t_hard_s=a.t_hard)
            while True:
                o, r, te, tr, info = env.step(pol.act(env))
                if te or tr: break
            rows.append({"seed": s, "outcome": info["outcome"], "speed": info["mean_speed"], "route": info["route_completion"], "wh_km": info["energy_wh"] / max(info["distance_m"] / 1000, 1e-3)})
            print(name, rows[-1], flush=True)
        oc = [r["outcome"] for r in rows]
        out[name] = {"n": len(rows), **{k: oc.count(k) / len(rows) for k in ("success", "crash", "off_road", "timeout")},
                     "mean_speed": float(np.mean([r["speed"] for r in rows])), "mean_route": float(np.mean([r["route"] for r in rows])), "episodes": rows}
        print("SUMMARY", name, {k: v for k, v in out[name].items() if k != "episodes"}, flush=True)
    Path("bev_rl/adaptive/results", f"safe_teacher_ceiling_d{a.density}_soft{a.t_soft}_hard{a.t_hard}.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
