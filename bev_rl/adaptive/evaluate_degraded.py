"""Evaluation under CALIBRATED perception degradation (degraded_env.py), FRESH seeds, paired across arms. One harness for every arm.

Arms (checkpoints under bev_rl/adaptive/runs unless noted):
  ov_split        clean-trained DAgger student (overlay BEV, 3 scalars, no weather input)              <- the model that was flat under the dropout-only proxy
  repo_bc_sparse / repo_bc_clear   behaviour-cloned baselines (GitHub BEV, no overlay, 6 scalars)
  deg_base        DAgger student trained UNDER the degradation (SafeTeacher labels), no weather input
  deg_ada         same with the weather branch (adaptive DAgger)
  ppo_plain / ppo_ada   PPO (reward v3) from deg_base, without / with the weather branch
  safe_teacher    rule-based reference on GROUND TRUTH (unaffected by perception; shows the ceiling)
Per episode: outcome, mean speed, Wh/km, route completion, minimum bumper gap to a vehicle ahead (privileged safety margin), steps.
Resumable per cell: results/degraded_eval/<tag>/<arm>__<weather>.json ; <cell>.partial.json every 20 episodes.
    python -m bev_rl.adaptive.evaluate_degraded --arms ov_split deg_base ppo_plain ppo_ada --n 200 --seed0 5000 --tag cal_d0.05
"""
import argparse
import json
from pathlib import Path

import numpy as np

WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
CKPT = {"ov_split": "bev_rl/adaptive/runs/ov_split/best.pt", "deg_base": "bev_rl/adaptive/runs/deg_base/best.pt", "deg_ada": "bev_rl/adaptive/runs/deg_ada/best.pt",
        "ppo_plain": "bev_rl/adaptive/runs/ppo_plain/best.pt", "ppo_ada": "bev_rl/adaptive/runs/ppo_ada/best.pt"}
REPO = {"repo_bc_sparse": "bev_policy/checkpoints/bc_sparse_traffic_5eps.pt", "repo_bc_clear": "bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt"}
ALL = list(CKPT) + list(REPO) + ["safe_teacher"]


def wname(w): return f"{w[0]}_{w[1]}"


def row(s, i):
    return {"seed": s, "outcome": i["outcome"], "speed": float(i["mean_speed"]), "wh_per_km": float(i["energy_wh"] / max(i["distance_m"] / 1000.0, 1e-3)),
            "route": float(i["route_completion"]), "min_gap_m": float(i.get("min_gap_m", 30.0)), "steps": int(i["steps"])}


def summarise(arm, w, a, rows):
    oc = [r["outcome"] for r in rows]; n = len(oc)
    return {"arm": arm, "weather": f"{w[0]}:{w[1]}", "seed0": a.seed0, "n": n, "density": a.density, "mode": a.mode,
            **{k: oc.count(k) / n for k in ("success", "crash", "off_road", "timeout")}, "mean_speed": float(np.mean([r["speed"] for r in rows])),
            "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])), "route": float(np.mean([r["route"] for r in rows])),
            "min_gap_m": float(np.mean([r["min_gap_m"] for r in rows])), "episodes": rows}


def run_cell(arm, w, a, env, torch, part):
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive.caution import KEEP
    from bev_rl.adaptive.degraded_env import make_safe_expert, run_episode
    from bev_rl.adaptive.env import SCALE
    rows = json.loads(part.read_text()) if part.exists() else []; done = {r["seed"] for r in rows}
    if arm in CKPT:
        ck = torch.load(CKPT[arm], map_location="cpu", weights_only=True); wd = int(ck.get("weather_dim", 0))
        net = BEVScalarPolicy(1, 6, wd); net.load_state_dict(ck["model_state"]); net.eval(); mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
    elif arm in REPO:
        ck = torch.load(REPO[arm], map_location="cpu", weights_only=True); net = BEVScalarPolicy(int(ck["in_channels"])); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32)
    for s in range(a.seed0, a.seed0 + a.n):
        if s in done: continue
        if arm in CKPT:
            *_, i = run_episode(env, net, mu, sd, torch, s, w, wd == 4)
        else:
            o, _ = env.reset(options={"scenario_seed": s, "weather": w}); env.drng = np.random.default_rng(s * 7 + 1); gap = 30.0
            pol = make_safe_expert(env) if arm == "safe_teacher" else None
            while True:
                if arm == "safe_teacher":
                    act = pol.act(env)
                else:                                                              # repo models: raw GitHub BEV (degraded objects), no overlay, all 6 scalars
                    bev = env.asm.build_bev_grid(env.degrade(env.truth_objects())); sc = ((o[:6] * SCALE - mu) / sd)[None].astype(np.float32)
                    with torch.no_grad():
                        act = net(torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc)).squeeze(0).numpy()
                o, r, te, tr, i = env.step(act); gap = min(gap, env.min_gap_m())
                if te or tr: i["min_gap_m"] = gap; break
        rows.append(row(s, i)); print(arm, wname(w), rows[-1], flush=True)
        if len(rows) % 20 == 0: part.write_text(json.dumps(rows))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--arms", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--mode", default="calibrated", choices=["calibrated", "dropout", "none"]); ap.add_argument("--tag", default="cal_d0.05")
    ap.add_argument("--only-weather", nargs="+", default=None, choices=["clear_0.0", "fog_1.0", "rain_0.6", "rain_1.0"])
    a = ap.parse_args(); OUT = Path("bev_rl/adaptive/results/degraded_eval") / a.tag; OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from bev_rl.adaptive.degraded_env import DegradedEnv
    for arm in a.arms:
        env = None
        for w in WEATHER:
            if a.only_weather and wname(w) not in a.only_weather: continue
            cell = OUT / f"{arm}__{wname(w)}.json"
            if cell.exists(): continue
            if env is None: env = DegradedEnv(range(a.seed0, a.seed0 + a.n), WEATHER, mode=a.mode, traffic_density=a.density, horizon=800, sample_seed=7)
            rows = run_cell(arm, w, a, env, torch, OUT / f"{arm}__{wname(w)}.partial.json")
            cell.write_text(json.dumps(summarise(arm, w, a, rows), indent=1)); (OUT / f"{arm}__{wname(w)}.partial.json").unlink(missing_ok=True)
            print("SUMMARY", arm, wname(w), {k: v for k, v in summarise(arm, w, a, rows).items() if k != "episodes"}, flush=True)
        if env is not None: env.close()


if __name__ == "__main__":
    main()
