"""Evaluate TEACHERS on their own (simulator only; teachers read the exact simulator state, no camera is involved).

Teachers
    stock_idm   MetaDrive IDM (30 km/h, 10 m gap), lane change as in the original OV-Split training          (reference)
    safe        SafeTeacher: IDM with lane change OFF + predictive safety layer (t_soft 2.4 s, t_hard 1.3 s)   (Teacher A, for OV-Split Camera)
    caut_idm    the ORIGINAL caution teacher: IDM with target speed 30(1-c/3) km/h and gap 10+15c m           (reference)
    cs0         Caution-SafeTeacher: caution IDM, lane change OFF, SafeTeacher layer with FIXED thresholds 2.4 / 1.3 s
    cs1         Caution-SafeTeacher: as cs0 but thresholds grow with caution: t_soft 2.4+1.0c, t_hard 1.3+0.5c
c = c_auto(weather) from the measured retention table (the same rule the caution models use). Weather has no other effect on a privileged teacher.

Per episode it records: outcome, steps (completion time = steps/10 s for successes), mean speed, Wh/km, fraction of steps stopped (< 0.5 m/s), fraction of steps in full brake,
free-flow mean speed (steps with no predicted conflict) and the commanded target speed, minimum predicted time-to-collision.

    python -m bev_rl.adaptive.teacher_eval --teacher cs0 --weather rain:1.0 --seeds 3100-3199 --horizon 1600 --out bev_rl/adaptive/results/teacher_validation
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def curvature_cap_mps(env, v_now, a_lat, v_min=3.5):
    """Curve guard (teacher side, privileged HD-map geometry): v_cap = sqrt(a_lat / max|curvature|) over the next max(15 m, 3 s) of route, never below v_min."""
    from bev_rl.adaptive.spec_bev import _lane_chain
    ag = env.md.agent; lanes = _lane_chain(ag.navigation); s0, _ = lanes[0].local_coordinates(ag.position)
    horizon = max(15.0, 3.0 * v_now); travelled, kmax = 0.0, 0.0
    for k, lane in enumerate(lanes):
        start = s0 if k == 0 else 0.0; end = min(float(lane.length), start + (horizon - travelled))
        if end > start + 1.0:
            ss = np.arange(start, end, 1.0); th = np.array([lane.heading_theta_at(float(x)) for x in ss])
            kmax = max(kmax, float(np.abs(np.angle(np.exp(1j * np.diff(th)))).max()))
        travelled += max(0.0, end - start)
        if travelled >= horizon - 1e-6: break
    return float("inf") if kmax < 1e-6 else max(v_min, float(np.sqrt(a_lat / kmax)))


def build_teacher(name, env, c):
    guard = None
    if "_g" in name: name, g = name.rsplit("_g", 1); guard = float(g)
    out = _build(name, env, c)
    return (*out, guard)


def _build(name, env, c):
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.caution import expert_params
    from bev_rl.adaptive.safe_teacher import SafeTeacher
    idm = IDMPolicy(env.md.agent, env.md.engine.global_seed)
    target = 30.0 / 3.6
    if name == "stock_idm": return idm, None, target
    if name == "caut_idm":
        idm.NORMAL_SPEED, idm.DISTANCE_WANTED = expert_params(c); idm.target_speed = idm.NORMAL_SPEED; return idm, None, idm.NORMAL_SPEED / 3.6
    idm.enable_lane_change = False
    if name == "safe": return idm, SafeTeacher(idm), target
    idm.NORMAL_SPEED, idm.DISTANCE_WANTED = expert_params(c); idm.target_speed = idm.NORMAL_SPEED
    if name == "cs0": return idm, SafeTeacher(idm), idm.NORMAL_SPEED / 3.6
    if name == "cs1": return idm, SafeTeacher(idm, t_soft_s=2.4 + 1.0 * c, t_hard_s=1.3 + 0.5 * c), idm.NORMAL_SPEED / 3.6
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--teacher", required=True, help="stock_idm | safe | caut_idm | cs0 | cs1, optionally with a curvature guard suffix _g<a_lat m/s^2>, e.g. safe_g3.0"); ap.add_argument("--weather", required=True)
    ap.add_argument("--seeds", default="3100-3199"); ap.add_argument("--horizon", type=int, default=1600); ap.add_argument("--density", type=float, default=0.05); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); kind, level = a.weather.split(":"); w = (kind, float(level)); lo, hi = map(int, a.seeds.split("-"))
    a.out.mkdir(parents=True, exist_ok=True); f = a.out / f"{a.teacher}__{kind}_{level}__d{a.density}__h{a.horizon}__s{lo}-{hi}.json"; part = f.with_suffix(".partial.json")
    if f.exists(): return
    from bev_rl.adaptive.caution import CautionEnv, c_auto
    env = CautionEnv(range(lo, hi + 1), [w], dropout=False, traffic_density=a.density, horizon=a.horizon, sample_seed=7)
    c = c_auto(kind, w[1], env.table)
    rows = json.loads(part.read_text()) if part.exists() else []; done = {r["seed"] for r in rows}
    for seed in range(lo, hi + 1):
        if seed in done: continue
        t0 = time.time(); env.reset(options={"scenario_seed": seed, "weather": w}); idm, safe, target, guard = build_teacher(a.teacher, env, c)
        speeds, free, stopped, hard, min_ttc, guard_steps = [], [], 0, 0, None, 0; v_cmd = idm.NORMAL_SPEED
        while True:
            if guard is not None:
                cap = curvature_cap_mps(env, float(env.md.agent.speed), guard); idm.NORMAL_SPEED = idm.target_speed = min(v_cmd, cap * 3.6); guard_steps += cap * 3.6 < v_cmd - 0.5
            act = np.clip(np.asarray(safe.act(env) if safe else idm.act(), np.float32)[:2], -1, 1)
            ttc = safe.last_ttc if safe else None
            if ttc is not None: min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)
            o, r, te, tr, info = env.step(act); sp = float(env.md.agent.speed)
            speeds.append(sp); stopped += sp < 0.5; hard += bool(safe and ttc is not None and ttc < safe.t_hard)
            if ttc is None: free.append(sp)
            if te or tr: break
        n = len(speeds)
        rows.append({"seed": seed, "outcome": info["outcome"], "steps": n, "speed": float(np.mean(speeds)), "wh_per_km": float(info["energy_wh"] / max(info["distance_m"] / 1000.0, 1e-3)),
                     "route": float(info["route_completion"]), "stopped_frac": stopped / n, "hard_brake_frac": hard / n, "free_flow_speed": float(np.mean(free)) if free else None,
                     "target_speed": float(target), "guard_active_frac": guard_steps / n if guard is not None else None, "min_ttc": min_ttc, "c": c, "wall_s": round(time.time() - t0, 1)})
        part.write_text(json.dumps(rows))
    oc = [r["outcome"] for r in rows]; nn = len(rows)
    ok = [r for r in rows if r["outcome"] == "success"]
    f.write_text(json.dumps({"teacher": a.teacher, "weather": a.weather, "c": c, "density": a.density, "horizon": a.horizon, "seeds": f"{lo}-{hi}", "n": nn,
                             **{k: oc.count(k) / nn for k in ("success", "crash", "off_road", "timeout")},
                             "completion_time_s": float(np.mean([r["steps"] / 10.0 for r in ok])) if ok else None, "mean_speed": float(np.mean([r["speed"] for r in rows])),
                             "wh_per_km": float(np.mean([r["wh_per_km"] for r in rows])), "stopped_frac": float(np.mean([r["stopped_frac"] for r in rows])),
                             "hard_brake_frac": float(np.mean([r["hard_brake_frac"] for r in rows])),
                             "free_flow_speed": float(np.mean([r["free_flow_speed"] for r in rows if r["free_flow_speed"] is not None])), "target_speed": float(target), "episodes": rows}, indent=1))
    part.unlink(missing_ok=True); print(a.teacher, a.weather, "success", oc.count("success") / nn, "n", nn, flush=True)
    env.close()


if __name__ == "__main__":
    main()
