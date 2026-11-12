"""PPO fine-tuning with reward v3 under CALIBRATED perception degradation, two variants from the SAME base checkpoint (so any difference is the weather input alone):

    python -m bev_rl.adaptive.ppo_adaptive --name ppo_plain --base bev_rl/adaptive/runs/deg_base/best.pt --variant plain
    python -m bev_rl.adaptive.ppo_adaptive --name ppo_ada   --base bev_rl/adaptive/runs/deg_base/best.pt --variant adaptive

plain   : network without weather input (weather_dim 0)                          -> the non-adaptive control
adaptive: identical function at step 0 + weather branch Linear(4->16) whose head columns start at 0 (ppo_bev.base_to_weather_ckpt); weather_vector(kind, intensity) as input
Both: student BEV = calibrated degraded BEV (degraded_env.py), reward = reward_v3 (privileged TTC/gap/energy terms), 'caution' hand rule NOT used.
Gate: the base itself (step 0) is a candidate; best = (validation success, -failure, later step). Every checkpoint is exported as a plain BEVScalarPolicy .pt (evaluate_degraded.py).
Seeds: train 2000-2999, validation 3100-3139. Weather mix and densities as in degraded_dagger.py.
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from bev_rl.adaptive.caution import KEEP
from bev_rl.adaptive.degraded_env import DegradedEnv
from bev_rl.adaptive.env import weather_vector
from bev_rl.adaptive.reward_v3 import RewardV3, step_reward_v3

TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.8), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3140))


class AdaptPPOEnv(DegradedEnv):
    def __init__(self, seeds, conditions, mean, std, adaptive, cfg=None, mode="calibrated", **kw):
        super().__init__(seeds, conditions, mode=mode, **kw)
        from bev_rl.adaptive.safe_teacher import SafeTeacher
        self.mean, self.std, self.adaptive, self.rcfg = np.asarray(mean, np.float32), np.asarray(std, np.float32), bool(adaptive), cfg or RewardV3()
        self._ttc = SafeTeacher(None, horizon_s=4.0)
        sp = {"bev": gym.spaces.Box(-1.0, 1.0, (1, 64, 64), np.float32), "scalar": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)}
        if self.adaptive: sp["weather"] = gym.spaces.Box(0.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Dict(sp)

    def _obs(self, info):
        raw = self.asm.extract_scalar_state(self.md, info)
        d = {"bev": self.student_bev()[None].astype(np.float32), "scalar": (((raw - self.mean) / self.std) * KEEP).astype(np.float32)}
        if self.adaptive: d["weather"] = weather_vector(*self.weather)
        return d

    def reset(self, *, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options); self._min_gap = 30.0
        return o, i

    def step(self, action):
        from bev_rl.energy import JOULES_PER_WH, step_work_j
        from bev_rl.telemetry import read_telemetry
        a = np.clip(np.asarray(action, np.float32), -1, 1); dt = self.rcfg.dt
        _, _, term, trunc, info = self.md.step(a); self.t += 1
        prev, tel = self.tel, read_telemetry(self.md)
        accel = (tel.speed_mps - prev.speed_mps) / dt; jerk = (accel - self.prev_accel) / dt
        if info.get("arrive_dest"): outcome = "success"
        elif any(info.get(k) for k in ("crash_vehicle", "crash_object", "crash_building", "crash_human")): outcome = "crash"
        elif info.get("out_of_road"): outcome = "off_road"
        else: outcome = None
        gap = self.min_gap_m(); self._min_gap = min(self._min_gap, gap)
        work = max(0.0, step_work_j(prev.speed_mps, tel.speed_mps, dt, self.energy_params)); wh = work / JOULES_PER_WH
        r, _ = step_reward_v3(self.rcfg, ds_m=tel.progress_m - prev.progress_m, speed_mps=tel.speed_mps, lateral_m=tel.lateral_m, heading_err_rad=tel.heading_err_rad,
                              action=a, prev_action=self.prev_action, jerk_mps3=jerk, ttc_s=self._ttc.ttc(self), gap_m=gap, wh=wh, outcome=outcome)
        self.energy_j += work; self.odometer += 0.5 * (prev.speed_mps + tel.speed_mps) * dt
        self.speeds.append(tel.speed_mps); self.lat.append(abs(tel.lateral_m))
        done = bool(term); tl = bool(trunc) or (self.t >= self.horizon and not done)
        self.tel, self.prev_action, self.prev_accel = tel, a, accel
        out = {"outcome": outcome or ("timeout" if tl else None), "route_completion": tel.route_completion, "energy_wh": self.energy_j / JOULES_PER_WH, "distance_m": self.odometer,
               "mean_speed": float(np.mean(self.speeds)), "steps": self.t, "min_gap_m": self._min_gap, "weather_kind": self.weather[0], "weather_level": float(self.weather[1]), "scenario": self.scenario}
        return self._obs(info), float(r), done, tl and not done, out


def export_actor(policy, mu, sd, wd, path, torch):
    """SB3 actor -> plain BEVScalarPolicy checkpoint (so evaluation uses the same code path as every other model)."""
    from bev_policy.model import BEVScalarPolicy
    bc = BEVScalarPolicy(1, 6, wd); fe = policy.pi_features_extractor
    bc.bev_encoder.load_state_dict(fe.bev_encoder.state_dict()); bc.scalar_encoder.load_state_dict(fe.scalar_encoder.state_dict())
    if wd: bc.weather_encoder.load_state_dict(fe.weather_encoder.state_dict())
    bc.action_head[0].load_state_dict(policy.mlp_extractor.policy_net[0].state_dict()); bc.action_head[2].load_state_dict(policy.action_net.state_dict())
    torch.save({"model_state": bc.state_dict(), "scalar_mean": list(map(float, mu)), "scalar_std": list(map(float, sd)), "in_channels": 1, "weather_dim": wd}, path)
    return bc


def make_env_fn(rank, a, ck):
    def f():
        import torch  # noqa
        from stable_baselines3.common.monitor import Monitor
        dens = [0.05, 0.05, 0.1][rank % 3]
        e = AdaptPPOEnv(range(2000, 3000), TRAIN_WEATHER, ck["scalar_mean"], ck["scalar_std"], a.variant == "adaptive", traffic_density=dens, horizon=600, sample_seed=a.seed * 100 + rank)
        return Monitor(e, info_keywords=("outcome",))
    return f


def validate(model, ck, a, n):
    env = AdaptPPOEnv(VAL_SEEDS, VAL_WEATHER, ck["scalar_mean"], ck["scalar_std"], a.variant == "adaptive", sequential=True, traffic_density=0.05, horizon=600)
    res = []
    for w in VAL_WEATHER:
        for s in VAL_SEEDS[:n]:
            o, _ = env.reset(options={"scenario_seed": s, "weather": w})
            while True:
                o, r, te, tr, i = env.step(model.predict(o, deterministic=True)[0])
                if te or tr: break
            res.append((i["outcome"] == "success", i["outcome"] in ("crash", "off_road")))
    env.close(); arr = np.array(res, float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--base", required=True); p.add_argument("--variant", choices=("plain", "adaptive"), required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--chunk", type=int, default=80_000); p.add_argument("--n-envs", type=int, default=3); p.add_argument("--val-n", type=int, default=12)
    a = p.parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from bev_rl.adaptive.ppo_bev import base_to_weather_ckpt
    from bev_rl.policy import BEVActorCritic, init_from_bc
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True); (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    ck = torch.load(a.base, weights_only=True); torch.manual_seed(a.seed); wd = 4 if a.variant == "adaptive" else 0
    init_ck = base_to_weather_ckpt(ck) if wd else {"in_channels": 1, "weather_dim": 0, "model_state": ck["model_state"]}
    venv = SubprocVecEnv([make_env_fn(i, a, ck) for i in range(a.n_envs)])
    model = PPO(BEVActorCritic, venv, learning_rate=a.lr, n_steps=256, batch_size=128, n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.0, target_kl=0.02,
                seed=a.seed, device="cpu", verbose=0, policy_kwargs={"log_std_init": -2.0})
    init_from_bc(model.policy, init_ck)
    actor = (list(model.policy.pi_features_extractor.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters()) + [model.policy.log_std])

    class Warm(BaseCallback):                       # critic-only for the first 25 rollouts (the critic starts from scratch; the actor must not be dragged by random advantages)
        def _on_training_start(self):
            self.n = 0
            for q in actor: q.requires_grad_(False)
        def _on_rollout_end(self):
            self.n += 1
            if self.n == 25:
                for q in actor: q.requires_grad_(True)
        def _on_step(self): return True
    warm = Warm(); t0, done, log = time.time(), 0, []
    v0 = validate(model, ck, a, a.val_n); best = (v0[0], v0[1], 0)
    export_actor(model.policy, ck["scalar_mean"], ck["scalar_std"], wd, out / "best.pt", torch); export_actor(model.policy, ck["scalar_mean"], ck["scalar_std"], wd, out / "step_0.pt", torch)
    log.append({"steps": 0, "val_success": v0[0], "val_fail": v0[1]}); print(json.dumps(log[-1]), flush=True)
    while done < a.steps:
        model.learn(a.chunk, reset_num_timesteps=False, callback=warm); done += a.chunk
        v = validate(model, ck, a, a.val_n); rec = {"steps": done, "val_success": v[0], "val_fail": v[1], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        export_actor(model.policy, ck["scalar_mean"], ck["scalar_std"], wd, out / f"step_{done}.pt", torch)
        if (v[0], -v[1], done) > (best[0], -best[1], best[2]):
            best = (v[0], v[1], done); export_actor(model.policy, ck["scalar_mean"], ck["scalar_std"], wd, out / "best.pt", torch); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))
    venv.close()


if __name__ == "__main__":
    main()
