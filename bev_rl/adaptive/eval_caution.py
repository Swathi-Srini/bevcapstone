"""Evaluation of the caution-conditioned policy on FRESH seeds (default 1010-1039; never used for training/validation).

Arms (same seeds, weather, density, horizon 800):
    caut_auto      caut_a,      c = c_auto(weather)   <- the proposed method
    caut_c0        caut_a,      c = 0                  (same net, caution off)
    caut_c1        caut_a,      c = 1                  (same net, always max caution)
    caut_c0_nodrop caut_a,      c = 0, perfect perception (upper bound; detector dropout OFF)
    ctrl           ctrl_nocaut  c = 0                  (trained without caution; isolates the effect of the caution mechanism)

    python -m bev_rl.adaptive.eval_caution --arm caut_auto --n 30
Writes bev_rl/adaptive/results/caution/<arm>.json (per-cell summary with Wilson 95% CIs, plus per-episode rows).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bev_rl.adaptive.caution import CautionEnv, c_auto, episode
from bev_rl.energy import JOULES_PER_WH  # noqa: F401

ARMS = {"caut_auto": ("caut_a", "auto", True), "caut_c0": ("caut_a", 0.0, True), "caut_c1": ("caut_a", 1.0, True),
        "caut_c0_nodrop": ("caut_a", 0.0, False), "ctrl": ("ctrl_nocaut", 0.0, True),
        "base_ovsplit": ("ov_split", 0.0, True), "base_ovsplit_nodrop": ("ov_split", 0.0, False)}   # previous best: DAgger on perfect perception, no weather/caution input
WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
SWEEPS = {"rain": [("clear", 0.0), ("rain", 0.1), ("rain", 0.2), ("rain", 0.3), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)],      # progressive degradation (RQ)
          "fog": [("clear", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 0.8), ("fog", 1.0)]}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - h)), float(min(1, c + h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS); ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1010); ap.add_argument("--density", type=float, default=0.2)
    ap.add_argument("--horizon", type=int, default=800)
    ap.add_argument("--sweep", choices=SWEEPS, default=None, help="progressive degradation sweep instead of the 4 fixed cells")
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    run, cmode, drop = ARMS[a.arm]
    ck = torch.load(f"bev_rl/adaptive/runs/{run}/best.pt", weights_only=True)
    if run == "ov_split":
        inner = BEVScalarPolicy(1, 6, 0); inner.load_state_dict(ck["model_state"]); inner.eval()
        net = lambda b, s_, c: inner(b, s_)            # baseline ignores caution
    else:
        net = BEVScalarPolicy(1, 6, 1); net.load_state_dict(ck["model_state"]); net.eval()
    mu, sd = np.array(ck["scalar_mean"], np.float32), np.array(ck["scalar_std"], np.float32)
    global WEATHER
    if a.sweep: WEATHER = SWEEPS[a.sweep]
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), WEATHER, dropout=drop, traffic_density=a.density, horizon=a.horizon, sample_seed=7)
    rows, cells = [], {}
    for w in WEATHER:
        c = c_auto(*w, env.table) if cmode == "auto" else float(cmode)
        outs = []
        for s in range(a.seed0, a.seed0 + a.n):
            *_, i = episode(env, net, mu, sd, torch, s, w, c)
            km = max(i["distance_m"] / 1000.0, 1e-3)
            r = {"weather": f"{w[0]}:{w[1]}", "seed": s, "c": c, "outcome": i["outcome"], "mean_speed": i["mean_speed"],
                 "wh_per_km": i["energy_wh"] / km, "steps": i["steps"], "route_completion": i["route_completion"]}
            rows.append(r); outs.append(r); print(json.dumps(r), flush=True)
        n = len(outs); succ = sum(o["outcome"] == "success" for o in outs); fail = sum(o["outcome"] in ("crash", "off_road") for o in outs)
        cells[f"{w[0]}:{w[1]}"] = {"c": c, "n": n, "success": succ / n, "success_ci": wilson(succ, n), "fail": fail / n, "fail_ci": wilson(fail, n),
                                   "timeout": sum(o["outcome"] == "timeout" for o in outs) / n, "mean_speed": float(np.mean([o["mean_speed"] for o in outs])),
                                   "wh_per_km": float(np.mean([o["wh_per_km"] for o in outs])), "route_completion": float(np.mean([o["route_completion"] for o in outs]))}
    env.close()
    out = Path("bev_rl/adaptive/results/caution"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.arm}{'_sweep_' + a.sweep if a.sweep else ''}.json").write_text(json.dumps({"arm": a.arm, "args": vars(a), "cells": cells, "episodes": rows}, indent=1))
    print(json.dumps(cells, indent=1))


if __name__ == "__main__":
    main()
