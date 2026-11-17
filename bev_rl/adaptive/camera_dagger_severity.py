"""MODEL B (oracle-caution reference policy) - camera-pipeline DAgger with KNOWN synthetic weather severity.   NEW FILE (2026-11-16, user authorisation "Model B training").

This is a thin wrapper around camera_dagger.py (which is NOT modified): it reuses make_env / load_net / student_act / teacher_driver / claim / train / select / freeze verbatim and only replaces
what depends on the RETIRED detector-retention caution:

    caution c            c = 0 (clear);  c = fog level (fog);  c = rain level (rain)          identity mapping, a documented design assumption (fog and rain levels are generator-normalised, NOT physically equal)
    teacher              SafeTeacher + curve guard 4.0 with caution (cs0_g4.0): target speed 30*(1-c/3) km/h, following distance 10+15c m   (unchanged teacher relationship)
    student input        the ORACLE c (Model B).  Model C later receives the image-estimated c at inference with the SAME network (plug-in design).
    init                 runs/caut_a/best.pt (read only; never written).  New output directory: results/camera_dagger_severity/e2e_sevB.  Model A / historical checkpoints are never touched.
    training weather     clear x2, fog 0.3 / 0.8 / 1.0, rain 0.2 / 0.6 / 1.0.  The evaluation probes (fog 0.6, rain 0.45) and the old probes (fog 0.45, rain 0.3) are NEVER trained on.
    episodes / round     12 (camera_dagger.py: 40; round 0 was launched with a 24-list and stopped after 12 episodes) - schedule decision, see the amendment; everything else of the recipe is unchanged: density 0.05, horizon 600, action noise 0.1, beta = 1 then 0.5^t,
                         Adam lr 1e-3 / batch 128 / MSE, epochs 18/6, retraining on ALL data so far.
    estimator data       every recorded step also stores the 7 image features of the final weather-processed front-left frame (F), so the severity estimator gets whole-route, density-0.05,
                         student/teacher-driven training frames at zero extra cost.  F never influences the policy.

    python -m bev_rl.adaptive.camera_dagger_severity freeze
    python -m bev_rl.adaptive.camera_dagger_severity collect  --round 0 --worker 1
    python -m bev_rl.adaptive.camera_dagger_severity train    --round 0
    python -m bev_rl.adaptive.camera_dagger_severity validate --round 2 --worker 1
    python -m bev_rl.adaptive.camera_dagger_severity select   --round 2
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from bev_rl.adaptive import camera_dagger as cd

ROOT = Path("bev_rl/adaptive/results/camera_dagger_severity")
NAME = "e2e_sevB"
EPISODES_PER_ROUND = int(os.environ.get("SEV_EPISODES", 12))      # decided 2026-11-16: 12 (measured 18.3 min / episode, one worker). Round 0 was started with 24 and stopped after 12 completed episodes.
TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.3), ("fog", 0.8), ("fog", 1.0), ("rain", 0.2), ("rain", 0.6), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3104))
VAL_HORIZON = 1600
cd.ROOT = ROOT                                                                  # camera_dagger.train/select/freeze/verify_frozen now work inside the new directory
cd.MODELS["sevB"] = dict(init="bev_rl/adaptive/runs/caut_a/best.pt", name=NAME, teacher="cs0_g4.0", weather=TRAIN_WEATHER, uses_c=True, epochs0=18, epochs=6,
                         val_weather=VAL_WEATHER, val_horizon=VAL_HORIZON)


def oracle_c(kind, level):
    """Identity caution mapping (user decision 2026-11-16): clear 0, fog -> fog level, rain -> rain level."""
    return 0.0 if kind == "clear" or level <= 0 else float(level)


class FeatureTap:
    """Plugged in as env.sensor: BuilderEnv.student_bev passes it the final weather-processed front-left frame exactly as it would the live sensor. It only RECORDS the seven features."""
    def __init__(self): self.F = []
    def reset(self): self.F = []
    def update(self, frame):
        from bev_rl.adaptive.visibility_sensor import features
        self.F.append(features(frame)); return 0.0


def round_tasks(rnd):
    out = []
    order = np.random.default_rng([rnd, 2026, 1116]).permutation(EPISODES_PER_ROUND)
    for k in range(EPISODES_PER_ROUND):
        rng = np.random.default_rng([rnd, 5, k, 31]); seed = int(rng.integers(2000, 3000))
        w = TRAIN_WEATHER[int(order[k]) % len(TRAIN_WEATHER)]                    # balanced: every entry 3x per round
        out.append((0.05, k, seed, w, oracle_c(*w)))
    return out


def collect(a):
    if a.worker > cd.MAX_WORKERS: return
    cd.verify_frozen()
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.env import SCALE
    cfg = cd.MODELS["sevB"]; rdir = ROOT / NAME / f"round{a.round}"; beta = 1.0 if a.round == 0 else 0.5 ** a.round
    net = mu = sd = None
    if a.round > 0: net, mu, sd = cd.load_net(ROOT / NAME / f"iter_{a.round - 1}.pt", True, torch)
    env = None
    for d, k, seed, w, c in round_tasks(a.round):
        name = f"d{d}_k{k}_s{seed}"
        if (rdir / f"{name}.npz").exists() or not cd.claim(rdir / "claims" / name): continue
        if env is None:
            env = cd.make_env(d, 600, a.round * 100 + a.worker, sensor=False); env.sensor = FeatureTap(); env.c_hat = 0.0
        rng = np.random.default_rng([a.round, int(round(d * 100)), k, 99]); t0 = time.time()
        o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1); env.sensor.reset()
        act_teacher = cd.teacher_driver(cfg["teacher"], env, c); B, S, Y, CH = [], [], [], []
        step = 0
        while True:
            teacher_drives = net is None or rng.random() < beta
            bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); ea = act_teacher(env)               # full camera pipeline EVERY step; label = clean expert action at this state
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); Y.append(ea); CH.append(c)                  # the student is given the ORACLE caution
            act = ea if teacher_drives else cd.student_act(net, True, torch, bev, raw, c, mu, sd)
            step += 1
            act = np.clip(act + rng.normal(0, a.noise, 2), -1, 1).astype(np.float32)
            o, r, te, tr, info = env.step(act)
            if te or tr: break
        rdir.mkdir(parents=True, exist_ok=True)
        F = np.asarray(env.sensor.F, np.float32); assert len(F) == len(B), (len(F), len(B))
        np.savez_compressed(rdir / f"{name}.npz", B=np.stack(B), S=np.stack(S), Y=np.stack(Y), C=np.asarray(CH, np.float32), F=F, c=np.float32(c), weather=f"{w[0]}:{w[1]}", kind=w[0], level=np.float32(w[1]),
                            density=d, seed=seed, outcome=str(info["outcome"]), teacher_drove=np.float32(beta))
        print(json.dumps({"round": a.round, "density": d, "seed": seed, "weather": f"{w[0]}:{w[1]}", "c": round(c, 2), "steps": step, "outcome": info["outcome"],
                          "energy_wh": round(info["energy_wh"], 2), "dist_m": round(info["distance_m"], 1), "wall_s": round(time.time() - t0)}), flush=True)
    if env is not None: env.md.close()


def validate(a):
    if a.worker > cd.MAX_WORKERS: return
    cd.verify_frozen()
    import torch
    torch.set_num_threads(2)
    from bev_rl.adaptive.env import SCALE
    run = ROOT / NAME; vdir = run / f"val_round{a.round}"
    net, mu, sd = cd.load_net(run / f"iter_{a.round}.pt", True, torch); env = None
    for w in VAL_WEATHER:
        for seed in VAL_SEEDS:
            name = f"{w[0]}_{w[1]}__s{seed}"
            if (vdir / f"{name}.json").exists() or not cd.claim(vdir / "claims" / name): continue
            if env is None: env = cd.make_env(0.05, VAL_HORIZON, 5000 + a.round, sensor=False)
            c = oracle_c(*w); t0 = time.time()
            o, _ = env.reset(options={"scenario_seed": seed, "weather": w}); env.drng = np.random.default_rng(seed * 7 + 1)
            while True:
                bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); o, r, te, tr, info = env.step(cd.student_act(net, True, torch, bev, raw, c, mu, sd))
                if te or tr: break
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"{name}.json").write_text(json.dumps({"seed": seed, "weather": f"{w[0]}:{w[1]}", "outcome": info["outcome"], "steps": int(info.get("steps", 0)), "energy_wh": info["energy_wh"],
                                                           "distance_m": info["distance_m"], "mean_speed": info["mean_speed"], "wall_s": round(time.time() - t0)}))
            print(name, info["outcome"], flush=True)
    if env is not None: env.md.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "train", "validate", "select", "freeze"]); ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--worker", type=int, default=1); ap.add_argument("--noise", type=float, default=0.1)
    a = ap.parse_args(); a.model = "sevB"
    {"collect": collect, "train": cd.train, "validate": validate, "select": cd.select, "freeze": cd.freeze}[a.cmd](a)
