"""PPO fine-tuning with a COMBINABLE energy + robustness reward, on the caution-conditioned BEV policy.

Start point: the previous best DAgger policy (ov_split, no weather/caution input) grafted with a caution input whose weights are
zero, so the PPO policy is function-identical to ov_split at step 0 (checked in main). PPO then learns to use c.
Env: CautionEnv (overlay BEV, measured detector dropout, constant friction = no weather physics). c = c_auto(weather).

Reward = bev_rl.reward.step_reward (progress/lateral/heading/smoothness/jerk/overspeed/terminal)
         - lambda_e * traction Wh/step                              [--lambda-e]       energy
         - w_vsafe * ((v - v_safe(c))+ / 2)^2  (+ progress credit capped at v_safe)    [--safe-speed]    visibility-coupled speed limit
         - w_head * max(0, 1 - gap / d_safe(c))^2   gap = nearest vehicle ahead (|x|<2.2 m, truth)   [--w-head]  caution-scaled headway
         crash/off-road penalty = crash_pen * (1 + crash_c * c)                                       [--crash-pen --crash-c]  risk-scaled
v_safe(c) = IDM target speed of the caution teacher (30 -> 20 km/h); d_safe(c) = 8 + 17 c metres.
Selection gate: best.zip = best validation (seeds 3100-3109, density 0.2, 4 weathers) INCLUDING the untouched start.

    python -m bev_rl.adaptive.ppo_caution --name rl_full --base bev_rl/adaptive/runs/ov_split/best.pt --lambda-e 0.05 --safe-speed --w-head 0.5
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from bev_rl.adaptive.caution import CautionEnv, KEEP, c_auto, expert_params
from bev_rl.adaptive.env import SCALE

TRAIN_WEATHER = [("clear", 0.0), ("clear", 0.0), ("fog", 0.6), ("fog", 1.0), ("rain", 0.2), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]
VAL_SEEDS = list(range(3100, 3110))


class RLCautionEnv(CautionEnv):
    def __init__(self, seeds, conditions, mean, std, safe_speed=False, w_head=0.0, w_vsafe=0.5, crash_c=0.0, **kw):
        super().__init__(seeds, conditions, **kw)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.safe_speed, self.w_head, self.w_vsafe, self.crash_c = safe_speed, w_head, w_vsafe, crash_c
        self.base_cfg = self.cfg; self.c = 0.0
        self.observation_space = gym.spaces.Dict({"bev": gym.spaces.Box(-1.0, 1.0, (1, 64, 64), np.float32),
                                                  "scalar": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
                                                  "weather": gym.spaces.Box(0.0, 1.0, (1,), np.float32)})

    def _obs(self, info):
        raw = self.asm.extract_scalar_state(self.md, info)
        return {"bev": self.student_bev()[None].astype(np.float32), "scalar": (((raw - self.mean) / self.std) * KEEP).astype(np.float32),
                "weather": np.array([self.c], np.float32)}

    def reset(self, *, seed=None, options=None):
        o = options or {}
        # weather is chosen inside AdaptiveEnv.reset; c must be known before the first _obs, so choose it here via a one-step lookahead
        self.c = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self.c = c_auto(*self.weather, self.table)
        v_safe = expert_params(self.c)[0] / 3.6
        crash = self.base_cfg.crash_penalty * (1.0 + self.crash_c * self.c)
        self.cfg = dataclasses.replace(self.base_cfg, crash_penalty=crash, offroad_penalty=crash,
                                       **({"v_cap_mps": v_safe, "w_overspeed": self.w_vsafe} if self.safe_speed else {}))
        return self._fix_obs(obs), info

    def _fix_obs(self, obs):
        obs = dict(obs); obs["weather"] = np.array([self.c], np.float32); return obs

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        if self.w_head > 0:
            gap = 17.5
            for o in self.truth_objects():
                if abs(o.x_right_m) < 2.2 and o.y_forward_m > 0:
                    gap = min(gap, o.y_forward_m - 0.5 * o.length_m - 2.25)
            d_safe = 8.0 + 17.0 * self.c
            r -= self.w_head * max(0.0, 1.0 - max(gap, 0.0) / d_safe) ** 2
        return obs, r, te, tr, info


def graft(ck):
    """ov_split (weather_dim 0) -> weather_dim 1 network with identical function (zero head columns for the caution branch)."""
    import torch
    from bev_policy.model import BEVScalarPolicy
    base = BEVScalarPolicy(1, 6, 0); base.load_state_dict(ck["model_state"])
    new = BEVScalarPolicy(1, 6, 1)
    new.bev_encoder.load_state_dict(base.bev_encoder.state_dict()); new.scalar_encoder.load_state_dict(base.scalar_encoder.state_dict())
    with torch.no_grad():
        W = torch.zeros(128, 160 + 16); W[:, :160] = base.action_head[0].weight; new.action_head[0].weight.copy_(W)
        new.action_head[0].bias.copy_(base.action_head[0].bias); new.action_head[2].load_state_dict(base.action_head[2].state_dict())
    return {"in_channels": 1, "weather_dim": 1, "model_state": new.state_dict()}, base


def validate(model, ck, a):
    env = RLCautionEnv(VAL_SEEDS, VAL_WEATHER, ck["scalar_mean"], ck["scalar_std"], traffic_density=0.2, horizon=600, sequential=True)
    res = []
    for w in VAL_WEATHER:
        for s in VAL_SEEDS[:a.val_n]:
            o, _ = env.reset(options={"scenario_seed": s, "weather": w})
            while True:
                o, r, te, tr, i = env.step(model.predict(o, deterministic=True)[0])
                if te or tr:
                    break
            res.append((i["outcome"] == "success", i["outcome"] in ("crash", "off_road"), i["energy_wh"] / max(i["distance_m"] / 1000, 1e-3)))
    env.close()
    r = np.array(res, float)
    return float(r[:, 0].mean()), float(r[:, 1].mean()), float(r[:, 2].mean())


def make_env_fn(rank, a, ck):
    def f():
        from stable_baselines3.common.monitor import Monitor
        e = RLCautionEnv(range(2000, 3000), TRAIN_WEATHER, ck["scalar_mean"], ck["scalar_std"], a.safe_speed, a.w_head, a.w_vsafe, a.crash_c,
                         traffic_density=[0.1, 0.2, 0.2][rank % 3], horizon=600, sample_seed=a.seed * 100 + rank,
                         lambda_e=a.lambda_e, crash_pen=a.crash_pen, v_cap=a.v_cap)
        return Monitor(e, info_keywords=("outcome",))
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--base", default="bev_rl/adaptive/runs/ov_split/best.pt")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lambda-e", type=float, default=0.0)
    p.add_argument("--crash-pen", type=float, default=50.0); p.add_argument("--crash-c", type=float, default=0.0)
    p.add_argument("--safe-speed", action="store_true"); p.add_argument("--w-vsafe", type=float, default=0.5)
    p.add_argument("--w-head", type=float, default=0.0); p.add_argument("--v-cap", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--chunk", type=int, default=30_000); p.add_argument("--n-envs", type=int, default=2)
    p.add_argument("--val-n", type=int, default=10)
    a = p.parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from bev_rl.policy import BEVActorCritic, init_from_bc
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    ck = torch.load(a.base, weights_only=True); torch.manual_seed(a.seed)
    gck, base = graft(ck)
    venv = SubprocVecEnv([make_env_fn(i, a, ck) for i in range(a.n_envs)])
    model = PPO(BEVActorCritic, venv, learning_rate=a.lr, n_steps=256, batch_size=128, n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.1,
                ent_coef=0.0, target_kl=0.02, seed=a.seed, device="cpu", verbose=0, policy_kwargs={"log_std_init": -2.0})
    init_from_bc(model.policy, gck)
    actor = (list(model.policy.pi_features_extractor.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters())
             + list(model.policy.action_net.parameters()) + [model.policy.log_std])

    class Warm(BaseCallback):                      # critic warm-up: actor frozen for the first 25 rollouts
        def _on_training_start(self):
            self.n = 0
            for q in actor: q.requires_grad_(False)
        def _on_rollout_end(self):
            self.n += 1
            if self.n == 25:
                for q in actor: q.requires_grad_(True)
        def _on_step(self): return True
    # step-0 equivalence check: PPO mean action == ov_split action (caution input is a no-op at init)
    e = RLCautionEnv(VAL_SEEDS, VAL_WEATHER, ck["scalar_mean"], ck["scalar_std"], traffic_density=0.2, horizon=50)
    o, _ = e.reset(options={"scenario_seed": 3100, "weather": ("rain", 1.0)})
    with torch.no_grad():
        pa = model.predict(o, deterministic=True)[0]
        mu_, sd_ = np.array(ck["scalar_mean"], np.float32), np.array(ck["scalar_std"], np.float32)
        ba = base(torch.from_numpy(o["bev"][None]), torch.from_numpy(o["scalar"][None])).numpy()[0]
    e.close(); print("step-0 |PPO - ov_split| =", float(np.abs(pa - ba).max()), flush=True)
    warm = Warm(); log, t0, done = [], time.time(), 0
    v0 = validate(model, ck, a); best = v0; model.save(out / "best.zip")
    log.append({"steps": 0, "val_success": v0[0], "val_fail": v0[1], "wh_per_km": v0[2]}); print(json.dumps(log[-1]), flush=True)
    while done < a.steps:
        model.learn(a.chunk, reset_num_timesteps=False, callback=warm); done += a.chunk
        v = validate(model, ck, a)
        rec = {"steps": done, "val_success": v[0], "val_fail": v[1], "wh_per_km": v[2], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True); model.save(out / f"ckpt_{done}.zip")
        if (v[0], -v[1]) > (best[0], -best[1]):
            best = v; model.save(out / "best.zip"); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))
    venv.close()


if __name__ == "__main__":
    main()
