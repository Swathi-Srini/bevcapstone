"""PAIRED CAMERA-PIPELINE EVALUATION of Model A / B / C with full instrumentation.   NEW FILE (2026-11-16).  Reuses the live pipeline (camera_dagger.make_env = the genuine BuilderEnv) and bev_rl/energy.py (unchanged, via env.step).

    A   existing non-adaptive OV-Split camera student     camera_dagger_final/e2e_ov_split_final/best.pt       no caution input
    B   oracle-caution reference                           camera_dagger_severity/e2e_sevB/best.pt              c = identity(true kind, true level)
    C   proposed system (plug-in, SAME network as B)       camera_dagger_severity/e2e_sevB/best.pt              c = image-estimated caution (SeverityEstimator on the weathered front-left frame)

Every arm is run on the same scenario seed with the same per-step weather noise (the noise seed is a function of the scenario and the step only), density 0.05, no teacher, no action noise.
The SeverityEstimator runs in SHADOW mode for every arm (A and B ignore its output) so that estimated caution is logged against the true caution everywhere.
Per episode, one JSON in <out>/episodes: outcome (success / crash / off_road / timeout), steps (duration = steps * 0.1 s), route completion, energy_wh and distance_m (energy.py), wh_per_km, mean speed,
and per-step arrays: speed, throttle/brake command, steer, c_true, c_hat, kind_hat, approx. bumper gap to the nearest vehicle ahead in the ego lane (PRIVILEGED simulator truth, logging only), cumulative energy.
Horizon 1600 is primary; horizon-800 results are derived from the same traces in the analysis (an episode still running at step 800 counts as timeout at 800).
    python -m bev_rl.adaptive.evaluate_severity_policies --out bev_rl/adaptive/results/severity_eval --tasks "clear:0.0@5200-5219;fog:1.0@5200-5219;rain:1.0@5200-5219" --arms A,B,C --worker 1
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bev_rl.adaptive import camera_dagger as cd

A_CKPT = "bev_rl/adaptive/results/camera_dagger_final/e2e_ov_split_final/best.pt"
B_CKPT = "bev_rl/adaptive/results/camera_dagger_severity/e2e_sevB/best.pt"
EGO_LEN = 4.5


def parse_tasks(s):
    out = []
    for part in s.split(";"):
        w, seeds = part.split("@"); kind, level = w.split(":"); lo, hi = (seeds.split("-") + [seeds])[:2]
        out += [((kind, float(level)), seed) for seed in range(int(lo), int(hi) + 1)]
    return out


def oracle_c(kind, level): return 0.0 if kind == "clear" or level <= 0 else float(level)


def lead_gap(env):
    """Approx. bumper gap (m) to the nearest vehicle ahead in the ego lane; None if none within the 17.5 m BEV window. Privileged truth - logged, never used for control."""
    try:
        g = [o.y_forward_m - 0.5 * (o.length_m + EGO_LEN) for o in env.truth_objects() if abs(o.x_right_m) < 1.6 and o.y_forward_m > 0]
        return float(min(g)) if g else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--tasks", required=True); ap.add_argument("--arms", default="A,B,C"); ap.add_argument("--worker", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=1600); ap.add_argument("--estimator", type=Path, default=Path("bev_rl/adaptive/results/severity_estimator/estimator.json")); a = ap.parse_args()
    if a.worker > cd.MAX_WORKERS: return
    cd.verify_frozen()
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.severity_estimator import SeverityEstimator
    arms = a.arms.split(","); nets = {}
    for arm in arms:
        if arm == "A": nets[arm] = cd.load_net(A_CKPT, False, torch)
        elif "sev" not in nets: nets["sev"] = cd.load_net(B_CKPT, True, torch)
    for d in ("episodes", "claims"): (a.out / d).mkdir(parents=True, exist_ok=True)
    env = cd.make_env(0.05, a.horizon, 77, sensor=False); env.sensor = SeverityEstimator(a.estimator); env.c_hat = 0.0
    for (kind, level), seed in parse_tasks(a.tasks):
        for arm in arms:
            name = f"{arm}__{kind}_{level}__s{seed}"; res = a.out / "episodes" / f"{name}.json"
            if res.exists() or not cd.claim(a.out / "claims" / name): continue
            with open(a.out / "claims" / f"_w{a.worker}.log", "a") as fh: fh.write(name + "\n")
            net, mu, sd = nets["A" if arm == "A" else "sev"]; c_true = oracle_c(kind, level); t0 = time.time()
            o, _ = env.reset(options={"scenario_seed": seed, "weather": (kind, level)}); env.drng = np.random.default_rng(seed * 7 + 1); env.sensor.reset(); env.c_hat = 0.0
            tr_ = {k: [] for k in ("speed", "throttle", "steer", "c_hat", "kind_hat", "gap", "energy_wh")}
            while True:
                bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)                    # student_bev also advances the estimator on the live weathered front-left frame
                c_in = 0.0 if arm == "A" else (c_true if arm == "B" else float(env.c_hat))
                act = cd.student_act(net, arm != "A", torch, bev, raw, c_in, mu, sd)
                tr_["speed"].append(float(env.md.agent.speed)); tr_["c_hat"].append(float(env.c_hat)); tr_["kind_hat"].append(int(env.sensor.kind_hat)); tr_["gap"].append(lead_gap(env))
                tr_["throttle"].append(float(act[1])); tr_["steer"].append(float(act[0]))
                o, r, te, tr, info = env.step(act); tr_["energy_wh"].append(float(info["energy_wh"]))
                if te or tr: break
            dist_km = max(info["distance_m"] / 1000.0, 1e-3); gaps = [g for g in tr_["gap"] if g is not None]
            out = {"arm": arm, "weather": f"{kind}:{level}", "kind": kind, "level": level, "seed": seed, "c_true": c_true, "outcome": info["outcome"], "steps": int(info["steps"]), "duration_s": info["steps"] * 0.1,
                   "route_completion": float(info["route_completion"]), "energy_wh": float(info["energy_wh"]), "distance_m": float(info["distance_m"]), "wh_per_km": float(info["energy_wh"] / dist_km),
                   "mean_speed": float(info["mean_speed"]), "min_gap_m": float(min(gaps)) if gaps else None, "horizon": a.horizon, "density": 0.05, "wall_s": round(time.time() - t0), "trace": tr_}
            res.write_text(json.dumps(out)); print(json.dumps({k: v for k, v in out.items() if k != "trace"}), flush=True)
    env.md.close()


if __name__ == "__main__":
    main()
