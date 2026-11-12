"""Evaluate teacher policies (and the IDM baseline) on reserved seeds across weather. 
python -m bev_rl.adaptive.evaluate --runs probe_C --out bev_rl/adaptive/results/probe  [--idm]"""
import argparse, csv, json
from pathlib import Path
import numpy as np

CONDS = [("clear", 0.0), ("fog", 1.0), ("rain", 0.3), ("rain", 0.45), ("rain", 0.6), ("rain", 0.8), ("rain", 1.0)]
SEEDS = list(range(1000, 1010))


def run_policy(name, act_fn, env, conds, seeds):
    rows = []
    for kind, lvl in conds:
        for s in seeds:
            o, _ = env.reset(options={"scenario_seed": s, "weather": (kind, lvl)})
            while True:
                o, r, te, tr, i = env.step(act_fn(o, env))
                if te or tr: break
            wh_km = i["energy_wh"] / (i["distance_m"] / 1000) if i["distance_m"] > 1 else float("nan")
            rows.append({"policy": name, "kind": kind, "level": lvl, "seed": s, "outcome": i["outcome"], "rc": i["route_completion"],
                         "energy_wh": i["energy_wh"], "dist_m": i["distance_m"], "wh_per_km": wh_km, "mean_speed": i["mean_speed"],
                         "mean_abs_lat": i["mean_abs_lat"], "steps": i["steps"], "mu": i["mu"]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[]); p.add_argument("--idm", action="store_true")
    p.add_argument("--ckpt", default="final"); p.add_argument("--out", type=Path, required=True); p.add_argument("--seeds", default=None)
    a = p.parse_args()
    import torch, ultralytics  # noqa
    from stable_baselines3 import PPO
    from bev_rl.adaptive.env import AdaptiveEnv
    a.out.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS if a.seeds is None else list(map(int, a.seeds.split(",")))
    rows = []
    for run in a.runs:
        cfg = json.loads((Path("bev_rl/adaptive/runs") / run / "config.json").read_text())
        env = AdaptiveEnv(seeds, CONDS, weather_input="zeroed" if cfg["mode"] == "unconditioned" else "explicit",
                          lambda_e=float(cfg["lambda_e"]), v_cap=float(cfg["v_cap"]), sequential=True)
        model = PPO.load(Path("bev_rl/adaptive/runs") / run / f"{a.ckpt}.zip", device="cpu")
        rows += run_policy(run, lambda o, e: model.predict(o, deterministic=True)[0], env, CONDS, seeds)
        env.close()
    if a.idm:
        from metadrive.policy.idm_policy import IDMPolicy
        import bev_rl.adaptive.env as E
        env = AdaptiveEnv(seeds, CONDS, sequential=True)
        env.md.close()
        from metadrive import MetaDriveEnv
        env.md = MetaDriveEnv({"use_render": False, "image_observation": False, "traffic_density": 0.0, "num_scenarios": 10000,
                               "start_seed": 10, "horizon": 800, "random_agent_model": False, "random_lane_width": True,
                               "random_lane_num": True, "out_of_route_done": True, "on_continuous_line_done": False,
                               "agent_policy": IDMPolicy, "log_level": 50})
        rows += run_policy("IDM_fixed", lambda o, e: [0.0, 0.0], env, CONDS, seeds)
        env.close()
    tag = "_".join(a.runs + (["idm"] if a.idm else []))[:80]
    with open(a.out / f"episodes_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    import collections
    g = collections.defaultdict(list)
    for r in rows: g[(r["policy"], r["kind"], r["level"])].append(r)
    for k, v in g.items():
        print("%-22s %-5s %.2f  succ %.2f  rc %.3f  Wh/km %6.1f  v %.2f  lat %.2f" % (k + (np.mean([x["outcome"] == "success" for x in v]),
              np.mean([x["rc"] for x in v]), np.nanmean([x["wh_per_km"] for x in v]), np.mean([x["mean_speed"] for x in v]), np.mean([x["mean_abs_lat"] for x in v]))))


if __name__ == "__main__":
    main()
