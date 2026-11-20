
from bev_rl.energy import wh_per_km_from_totals
"""PROTOTYPE of the Visibility-Conditioned SafeTeacher (simulator only, no camera). New file: nothing existing is modified.

    target speed = min( IDM target 30 km/h,  curve-guard cap (a_lat 4.0),  visibility cap )
    visibility cap  v_vis = -b*t_r + sqrt( (b*t_r)^2 + 2*b*(R - m) ),   floored at v_min      (from  v*t_r + v^2/(2b) <= R - m, classical stopping-distance kinematics)
    collision layer = SafeTeacher (constant-velocity TTC braking, unchanged)

R here is a FIXED ILLUSTRATIVE effective perception range per run (NOT measured, NOT estimated from images): this prototype only tests how the teacher behaves as R shrinks,
and that it reduces to Teacher A (SafeTeacher + guard 4.0) when R is large (clear weather). The measured R_eff and the image-based estimator are separate, not yet done.

    python -m bev_rl.adaptive.visibility_teacher --ranges inf,17.5,12,8,5 --seeds 3100-3149 --out bev_rl/adaptive/results/visibility_teacher_proto
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def v_visibility(R, t_r, b, m, v_min):
    if math.isinf(R): return float("inf")
    x = R - m
    if x <= 0: return v_min
    return max(v_min, -b * t_r + math.sqrt((b * t_r) ** 2 + 2 * b * x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ranges", default="inf,17.5,12,8,5"); ap.add_argument("--seeds", default="3100-3149"); ap.add_argument("--horizon", type=int, default=1600)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--tr", type=float, default=0.5); ap.add_argument("--b", type=float, default=4.0); ap.add_argument("--m", type=float, default=2.0)
    ap.add_argument("--vmin", type=float, default=3.0); ap.add_argument("--guard", type=float, default=4.0); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); lo, hi = map(int, a.seeds.split("-")); a.out.mkdir(parents=True, exist_ok=True)
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.teacher_eval import build_teacher, curvature_cap_mps
    env = CautionEnv(range(lo, hi + 1), [("clear", 0.0)], dropout=False, traffic_density=a.density, horizon=a.horizon, sample_seed=7)
    for R in [float(x) for x in a.ranges.split(",")]:
        f = a.out / f"vis_R{'inf' if math.isinf(R) else R}__tr{a.tr}_b{a.b}_m{a.m}_vmin{a.vmin}__d{a.density}__h{a.horizon}__s{lo}-{hi}.json"
        if f.exists(): continue
        v_vis = v_visibility(R, a.tr, a.b, a.m, a.vmin); rows = []
        for seed in range(lo, hi + 1):
            t0 = time.time(); env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); idm, safe, target, _ = build_teacher("safe", env, 0.0); v_cmd = idm.NORMAL_SPEED
            speeds, free, stopped, hard, bind, vbind, min_ttc = [], [], 0, 0, 0, 0, None
            while True:
                cap = curvature_cap_mps(env, float(env.md.agent.speed), a.guard); lim = min(v_cmd, cap * 3.6, v_vis * 3.6); idm.NORMAL_SPEED = idm.target_speed = lim
                bind += v_vis * 3.6 < min(v_cmd, cap * 3.6) - 0.05
                act = np.clip(np.asarray(safe.act(env), np.float32)[:2], -1, 1); ttc = safe.last_ttc
                if ttc is not None: min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
                o, r, te, tr, info = env.step(act); sp = float(env.md.agent.speed); speeds.append(sp); stopped += sp < 0.5; hard += bool(ttc is not None and ttc < safe.t_hard)
                if ttc is None: free.append(sp)
                if te or tr: break
            n = len(speeds)
            rows.append({"seed": seed, "outcome": info["outcome"], "steps": n, "speed": float(np.mean(speeds)), "wh_per_km": float(wh_per_km_from_totals(info["energy_wh"], info["distance_m"])),
                         "stopped_frac": stopped / n, "hard_brake_frac": hard / n, "free_flow_speed": float(np.mean(free)) if free else None, "vis_cap_binding_frac": bind / n, "min_ttc": min_ttc})
        oc = [r["outcome"] for r in rows]; n = len(rows); ok = [r for r in rows if r["outcome"] == "success"]
        out = {"R_m": None if math.isinf(R) else R, "v_vis_mps": None if math.isinf(v_vis) else v_vis, "t_r": a.tr, "b": a.b, "m": a.m, "v_min": a.vmin, "guard": a.guard, "density": a.density, "horizon": a.horizon, "seeds": f"{lo}-{hi}", "n": n,
               **{k: oc.count(k) / n for k in ("success", "crash", "off_road", "timeout")}, "success_80s": sum(r["outcome"] == "success" and r["steps"] <= 800 for r in rows) / n,
               "completion_time_s": float(np.mean([r["steps"] / 10.0 for r in ok])) if ok else None, "mean_speed": float(np.mean([r["speed"] for r in rows])), "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])),
               "stopped_frac": float(np.mean([r["stopped_frac"] for r in rows])), "hard_brake_frac": float(np.mean([r["hard_brake_frac"] for r in rows])),
               "free_flow_speed": float(np.mean([r["free_flow_speed"] for r in rows if r["free_flow_speed"] is not None])), "vis_cap_binding_frac": float(np.mean([r["vis_cap_binding_frac"] for r in rows])), "episodes": rows}
        f.write_text(json.dumps(out, indent=1)); print(f"R={R} v_vis={v_vis:.2f} success {out['success']:.2f} crash {out['crash']:.2f} off {out['off_road']:.2f} timeout {out['timeout']:.2f} speed {out['mean_speed']:.2f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
