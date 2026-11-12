"""Evaluate TWO-CUE pose-reliability policies (pose_reliability_dagger.py --cam-cue) on the localisation-error protocol (fresh seeds 1010-1049, density 0.05, clear,
persistent map pose error with random sign, SAME error draws as the baseline sweep) crossed with two camera-detector noise levels (Lcam 0.33 = 0.2 m/0.02 rad, 1.0 = 0.6 m/0.06 rad per-step sd).
Conditioned arms are told c = [min(1, lat/1.5), Lcam]; unconditioned get [0, 0]. Writes results/pose_two_cue_<arm>_d0.05.json.
    python -m bev_rl.adaptive.evaluate_pose_two_cue_policy --arms pose_cam_conditioned pose_cam_unconditioned --n 40
"""
import argparse, json
from pathlib import Path
import numpy as np
from bev_rl.adaptive.evaluate_localisation_error_sensitivity import LEVELS, wilson
from bev_rl.adaptive.pose_reliability_dagger import LAT_MAX, CAM_LAT, CAM_HEAD, KEEP_TWO_CUE, prep2

CAM_LEVELS = [0.33, 1.0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=1010); ap.add_argument("--density", type=float, default=0.05)
    a = ap.parse_args()
    import torch
    from bev_policy.model import BEVScalarPolicy
    from bev_rl.adaptive import spec_bev
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.env import SCALE
    env = CautionEnv(range(a.seed0, a.seed0 + a.n), [("clear", 0.0)], dropout=False, traffic_density=a.density, horizon=800, sample_seed=7)
    for arm in a.arms:
        ck = torch.load(f"bev_rl/adaptive/runs/{arm}/best.pt", map_location="cpu", weights_only=True); cond = bool(ck.get("condition", False))
        net = BEVScalarPolicy(1, 6, 2); net.load_state_dict(ck["model_state"]); net.eval()
        mu, sd = np.asarray(ck["scalar_mean"], np.float32), np.asarray(ck["scalar_std"], np.float32); cells, rows = {}, []
        for lcam in CAM_LEVELS:
            for dx, dth in LEVELS:
                res = []; cvec = [min(1.0, dx / LAT_MAX), lcam] if cond else [0.0, 0.0]
                for s in range(a.seed0, a.seed0 + a.n):
                    rng = np.random.default_rng(s * 13 + 5); sx, st = rng.choice([-1.0, 1.0], 2); nrng = np.random.default_rng(s * 17 + 3)
                    o, _ = env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)}); spec_bev.POSE_ERROR = (sx * dx, 0.0, st * dth)
                    while True:
                        bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); raw[2] += nrng.normal(0, lcam * CAM_LAT); raw[3] += nrng.normal(0, lcam * CAM_HEAD)
                        with torch.no_grad(): act = net(*prep2(torch, bev, raw, cvec, mu, sd, KEEP_TWO_CUE)).squeeze(0).numpy()
                        o, r, te, tr, i = env.step(act)
                        if te or tr: break
                    spec_bev.POSE_ERROR = None
                    res.append(i["outcome"]); rows.append({"arm": arm, "cam_level": lcam, "lat_m": dx, "heading_rad": dth, "seed": s, "outcome": i["outcome"], "route": i["route_completion"]})
                n = len(res); k = res.count("success"); key = f"cam{lcam}_lat{dx}_head{dth}"
                cells[key] = {"n": n, "cam_level": lcam, "lat_m": dx, "heading_rad": dth, "success": k / n, "success_ci": wilson(k, n), "crash": res.count("crash") / n, "off_road": res.count("off_road") / n, "timeout": res.count("timeout") / n, "c_fed": cvec}
                print(arm, key, {x: cells[key][x] for x in ("success", "crash", "off_road", "timeout")}, flush=True)
        Path("bev_rl/adaptive/results", f"pose_two_cue_{arm}_d{a.density}.json").write_text(json.dumps({"arm": arm, "cells": cells, "episodes": rows}, indent=1))

if __name__ == "__main__":
    main()
