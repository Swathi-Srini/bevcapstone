"""PPO fine-tuning of the DAgger base on the repo BEV (+ 6-D scalar + explicit weather), with measured detector dropout.

Policy side only. The BEV is built by the unmodified GitHub grid builder from simulator vehicles; under weather each vehicle is
kept with probability = measured YOLO detection retention (bev_rl/adaptive/results/retention_table.json, GitHub weather
augmentation on the GitHub detector, 20 paired scenes), capped at 1. Fog/rain add NO dynamics change.

    python -m bev_rl.adaptive.ppo_bev --name ppo_cond --base bev_rl/adaptive/runs/base_a/best.pt --mode conditioned
Gate: best.zip is the checkpoint with the best validation score INCLUDING the untouched base (step 0).
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from bev_rl.adaptive.env import AdaptiveEnv, SCALE, weather_vector

TRAIN_WEATHER = [("clear", 0.0), ("fog", 0.6), ("fog", 0.8), ("fog", 1.0), ("rain", 0.2), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
VAL_WEATHER = [("clear", 0.0), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6)]
VAL_SEEDS = list(range(3100, 3110))
RET_PATH = Path("bev_rl/adaptive/results/retention_table.json")


def keep_prob(kind, level, table):
    if kind == "clear" or level <= 0:
        return 1.0
    pts = sorted((float(k.split(":")[1]), min(1.0, v["retention"])) for k, v in table.items() if k.startswith(kind + ":"))
    xs, ys = [0.0] + [p[0] for p in pts], [1.0] + [p[1] for p in pts]
    return float(np.interp(level, xs, ys))


class TrafficBEVEnv(AdaptiveEnv):
    def __init__(self, seeds, conditions, mean, std, weather_input="explicit", traffic_density=0.1, dropout=True, overlay=False, mask=None, **kw):
        super().__init__(seeds, conditions, weather_input=weather_input, traffic_density=traffic_density, **kw)
        self.overlay = bool(overlay); self.mask = np.ones(6, np.float32) if mask is None else np.asarray(mask, np.float32)
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)
        self.table = json.loads(RET_PATH.read_text()); self.dropout = dropout
        self.drng = np.random.default_rng(kw.get("sample_seed", 0) + 17)
        self.observation_space = gym.spaces.Dict({"bev": gym.spaces.Box(-1.0, 1.0, (1, 64, 64), np.float32),
                                                  "scalar": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32),
                                                  "weather": gym.spaces.Box(0.0, 1.0, (4,), np.float32)})

    def _obs(self, info):
        raw = self.asm.extract_scalar_state(self.md, info)
        p = keep_prob(*self.weather, self.table) if self.dropout else 1.0
        objs = [o for o in self.truth_objects() if self.drng.random() < p]
        w = weather_vector(*self.weather) if self.weather_input == "explicit" else np.zeros(4, np.float32)
        grid = self.asm.build_bev_grid(objs)
        if self.overlay:
            from bev_rl.adaptive.spec_bev import overlay_github_bev
            grid = overlay_github_bev(grid, self.md)
        return {"bev": grid[None].astype(np.float32),
                "scalar": (((raw - self.mean) / self.std) * self.mask).astype(np.float32), "weather": w}


def mask_of(a):
    return np.array([1.0 if str(i) in a.scalar_keep.split(",") else 0.0 for i in range(6)], np.float32)


def make_env_fn(rank, a, ck):
    def f():
        import torch, ultralytics  # noqa
        from stable_baselines3.common.monitor import Monitor
        dens = [0.05, 0.1, 0.2][rank % 3]
        e = TrafficBEVEnv(range(2000, 3000), TRAIN_WEATHER, ck["scalar_mean"], ck["scalar_std"],
                          "zeroed" if a.mode == "unconditioned" else "explicit", dens, True, overlay=a.overlay, mask=mask_of(a),
                          lambda_e=a.lambda_e, crash_pen=a.crash_pen, v_cap=a.v_cap, horizon=600, sample_seed=a.seed * 100 + rank)
        return Monitor(e, info_keywords=("outcome",))
    return f


def base_to_weather_ckpt(ck):
    """Net surgery: the weather_dim=0 base -> weather_dim=4 network with identical function (extra head columns = 0)."""
    import torch
    from bev_policy.model import BEVScalarPolicy
    base = BEVScalarPolicy(1, 6, 0); base.load_state_dict(ck["model_state"])
    new = BEVScalarPolicy(1, 6, 4)
    new.bev_encoder.load_state_dict(base.bev_encoder.state_dict()); new.scalar_encoder.load_state_dict(base.scalar_encoder.state_dict())
    with torch.no_grad():
        W = torch.zeros(128, 176); W[:, :160] = base.action_head[0].weight; new.action_head[0].weight.copy_(W)
        new.action_head[0].bias.copy_(base.action_head[0].bias)
        new.action_head[2].load_state_dict(base.action_head[2].state_dict())
    return {"in_channels": 1, "weather_dim": 4, "model_state": new.state_dict()}


def validate(model, ck, a, torch):
    """Deterministic validation on seeds 3100-3109 (disjoint from train/eval) at density 0.1 and 0.2."""
    res = []
    for dens in (0.1, 0.2):
        env = TrafficBEVEnv(VAL_SEEDS, VAL_WEATHER, ck["scalar_mean"], ck["scalar_std"], "zeroed" if a.mode == "unconditioned" else "explicit",
                            dens, True, overlay=a.overlay, mask=mask_of(a), sequential=True, horizon=600, v_cap=a.v_cap, lambda_e=a.lambda_e, crash_pen=a.crash_pen)
        for w in VAL_WEATHER:
            for s in VAL_SEEDS[:a.val_n]:
                o, _ = env.reset(options={"scenario_seed": s, "weather": w})
                while True:
                    o, r, te, tr, i = env.step(model.predict(o, deterministic=True)[0])
                    if te or tr:
                        break
                res.append((i["outcome"] == "success", i["outcome"] in ("crash", "off_road")))
        env.close()
    arr = np.array(res, float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--base", required=True)
    p.add_argument("--mode", choices=("conditioned", "unconditioned"), default="conditioned")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lambda-e", type=float, default=0.0)
    p.add_argument("--crash-pen", type=float, default=50.0); p.add_argument("--v-cap", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--steps", type=int, default=600_000)
    p.add_argument("--chunk", type=int, default=60_000); p.add_argument("--n-envs", type=int, default=3)
    p.add_argument("--val-n", type=int, default=10, help="validation seeds per cell (smoke tests use fewer)")
    p.add_argument("--overlay", action="store_true"); p.add_argument("--scalar-keep", default="0,1,2,3,4,5")
    a = p.parse_args()
    import torch, ultralytics  # noqa
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from bev_rl.policy import BEVActorCritic, init_from_bc
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    ck = torch.load(a.base, weights_only=True)
    torch.manual_seed(a.seed)
    venv = SubprocVecEnv([make_env_fn(i, a, ck) for i in range(a.n_envs)])
    model = PPO(BEVActorCritic, venv, learning_rate=a.lr, n_steps=256, batch_size=128, n_epochs=5, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.0, target_kl=0.02, seed=a.seed, device="cpu", verbose=0, policy_kwargs={"log_std_init": -2.0})
    init_from_bc(model.policy, base_to_weather_ckpt(ck))
    actor = (list(model.policy.pi_features_extractor.parameters()) + list(model.policy.mlp_extractor.policy_net.parameters())
             + list(model.policy.action_net.parameters()) + [model.policy.log_std])

    class Warm(BaseCallback):
        def _on_training_start(self):
            self.n = 0
            for q in actor: q.requires_grad_(False)
        def _on_rollout_end(self):
            self.n += 1
            if self.n == 25:
                for q in actor: q.requires_grad_(True)
        def _on_step(self): return True
    warm = Warm()
    best, log, t0, done = None, [], time.time(), 0
    v0 = validate(model, ck, a, torch); best = v0
    model.save(out / "best.zip"); log.append({"steps": 0, "val_success": v0[0], "val_fail": v0[1]}); print(json.dumps(log[-1]), flush=True)
    while done < a.steps:
        model.learn(a.chunk, reset_num_timesteps=False, callback=warm); done += a.chunk
        v = validate(model, ck, a, torch)
        rec = {"steps": done, "val_success": v[0], "val_fail": v[1], "t_min": (time.time() - t0) / 60}
        log.append(rec); print(json.dumps(rec), flush=True)
        model.save(out / f"ckpt_{done}.zip")
        if (v[0], -v[1]) > (best[0], -best[1]):
            best = v; model.save(out / "best.zip"); (out / "best.json").write_text(json.dumps(rec))
        (out / "log.json").write_text(json.dumps(log, indent=1))
    venv.close()


if __name__ == "__main__":
    main()
