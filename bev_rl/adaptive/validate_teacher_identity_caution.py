
from bev_rl.energy import wh_per_km_from_totals
"""TEACHER B validation under the IDENTITY caution mapping (c = fog/rain level, 0 in clear).   NEW FILE (2026-11-16).

Same teacher as camera_dagger (SafeTeacher + caution IDM + curve guard 4.0, name cs0_g4.0), same loop as teacher_eval.py, but caution c = severity level instead of the retired c_auto(retention).
Fast BEV-level environment (CautionEnv, dropout=False: the teacher is privileged, no weather physics), density 0.05, seeds 3100-3149 (the camera_dagger validation seed range), horizon 1600.
Logs outcome, mean speed, Wh/km, hard-brake and stopped fractions. Does not modify anything; output goes to results/teacher_identity_caution/.
    python -m bev_rl.adaptive.validate_teacher_identity_caution --weather fog:1.0 --seeds 3100-3149
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weather", required=True); ap.add_argument("--seeds", default="3100-3149"); ap.add_argument("--horizon", type=int, default=1600)
    ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--out", type=Path, default=Path("bev_rl/adaptive/results/teacher_identity_caution")); a = ap.parse_args()
    kind, level = a.weather.split(":"); w = (kind, float(level)); lo, hi = map(int, a.seeds.split("-")); c = 0.0 if kind == "clear" else float(level)
    a.out.mkdir(parents=True, exist_ok=True); f = a.out / f"cs0_g4.0__{kind}_{level}__h{a.horizon}__s{lo}-{hi}.json"
    if f.exists(): return
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.teacher_eval import build_teacher, curvature_cap_mps
    env = CautionEnv(range(lo, hi + 1), [w], dropout=False, traffic_density=a.density, horizon=a.horizon, sample_seed=7); rows = []
    for seed in range(lo, hi + 1):
        t0 = time.time(); env.reset(options={"scenario_seed": seed, "weather": w}); idm, safe, target, guard = build_teacher("cs0_g4.0", env, c); v_cmd = idm.NORMAL_SPEED
        speeds, stopped, hard = [], 0, 0
        while True:
            cap = curvature_cap_mps(env, float(env.md.agent.speed), guard); idm.NORMAL_SPEED = idm.target_speed = min(v_cmd, cap * 3.6)
            act = np.clip(np.asarray(safe.act(env), np.float32)[:2], -1, 1); ttc = safe.last_ttc
            o, r, te, tr, info = env.step(act); sp = float(env.md.agent.speed); speeds.append(sp); stopped += sp < 0.5; hard += bool(ttc is not None and ttc < safe.t_hard)
            if te or tr: break
        n = len(speeds)
        rows.append({"seed": seed, "outcome": info["outcome"], "steps": n, "mean_speed": float(np.mean(speeds)), "energy_wh": float(info["energy_wh"]), "distance_m": float(info["distance_m"]),
                     "wh_per_km": float(wh_per_km_from_totals(info["energy_wh"], info["distance_m"])), "route": float(info["route_completion"]), "stopped_frac": stopped / n, "hard_brake_frac": hard / n,
                     "wall_s": round(time.time() - t0, 1)})
    oc = [r["outcome"] for r in rows]; n = len(rows)
    f.write_text(json.dumps({"teacher": "cs0_g4.0", "weather": a.weather, "c": c, "density": a.density, "horizon": a.horizon, "seeds": f"{lo}-{hi}", "n": n,
                             **{k: oc.count(k) for k in ("success", "crash", "off_road", "timeout")}, "mean_speed": float(np.mean([r["mean_speed"] for r in rows])),
                             "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])), "target_speed_mps": 30.0 * (1 - c / 3) / 3.6, "episodes": rows}, indent=1))
    print(a.weather, "c", c, {k: oc.count(k) for k in ("success", "crash", "off_road", "timeout")}, "speed", round(float(np.mean([r["mean_speed"] for r in rows])), 2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
