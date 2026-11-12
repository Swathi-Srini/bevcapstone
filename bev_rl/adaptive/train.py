"""PPO teacher for weather-adaptive energy-aware driving.  python -m bev_rl.adaptive.train --name C_s0 --seed 0 --mode conditioned --lambda-e 0"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

CONDS = {"clear": [("clear", 0.0)],
         "mixed": [("clear", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 1.0), ("rain", 0.3), ("rain", 0.6), ("rain", 1.0)]}


def make(rank, a):
    def f():
        import torch, ultralytics  # noqa  (DLL order)
        from bev_rl.adaptive.env import AdaptiveEnv
        from stable_baselines3.common.monitor import Monitor
        e = AdaptiveEnv(range(2000, 3000), CONDS[a.train_weather], weather_input="zeroed" if a.mode == "unconditioned" else "explicit",
                        lambda_e=a.lambda_e, crash_pen=a.crash_pen, sample_seed=a.seed * 100 + rank, v_cap=a.v_cap)
        return Monitor(e, info_keywords=("outcome", "route_completion", "energy_wh", "mu"))
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("conditioned", "unconditioned"), default="conditioned")
    p.add_argument("--train-weather", choices=tuple(CONDS), default="mixed")
    p.add_argument("--lambda-e", type=float, default=0.0); p.add_argument("--v-cap", type=float, default=12.0)
    p.add_argument("--init-bc", default="bev_rl/adaptive/bc_init.pt"); p.add_argument("--warmup-rollouts", type=int, default=20)
    p.add_argument("--crash-pen", type=float, default=50.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=1_000_000); p.add_argument("--n-envs", type=int, default=8)
    a = p.parse_args()
    import torch, ultralytics  # noqa
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    out = Path("bev_rl/adaptive/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    venv = SubprocVecEnv([make(i, a) for i in range(a.n_envs)])
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=512, n_epochs=10, learning_rate=a.lr, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.0, target_kl=0.03, seed=a.seed, device="cpu", verbose=0,
                policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128]), "log_std_init": -1.6})
    sd = torch.load(a.init_bc, weights_only=True)
    pol = model.policy
    pn = pol.mlp_extractor.policy_net
    pn[0].load_state_dict({"weight": sd["0.weight"], "bias": sd["0.bias"]}); pn[2].load_state_dict({"weight": sd["2.weight"], "bias": sd["2.bias"]})
    pol.action_net.load_state_dict({"weight": sd["4.weight"], "bias": sd["4.bias"]})
    from stable_baselines3.common.callbacks import BaseCallback
    actor = list(pn.parameters()) + list(pol.action_net.parameters()) + [pol.log_std]

    class Warm(BaseCallback):
        def _on_training_start(self):
            self.n = 0
            for q in actor: q.requires_grad_(False)
        def _on_rollout_end(self):
            self.n += 1
            if self.n == a.warmup_rollouts:
                for q in actor: q.requires_grad_(True)
        def _on_step(self): return True
    warm = Warm()
    t0 = time.time(); done = 0; chunk = 100_000
    log = []
    from bev_rl.adaptive.env import AdaptiveEnv
    VAL_SEEDS, VAL_CONDS = [3100, 3101, 3102, 3103, 3104], [("clear", 0.0), ("rain", 0.6), ("rain", 1.0)]
    val_env = AdaptiveEnv(VAL_SEEDS, VAL_CONDS, weather_input="zeroed" if a.mode == "unconditioned" else "explicit",
                          lambda_e=a.lambda_e, crash_pen=a.crash_pen, v_cap=a.v_cap, sequential=True)

    def validate():
        succ, rew = [], []
        for c in VAL_CONDS:
            for sd in VAL_SEEDS:
                o, _ = val_env.reset(options={"scenario_seed": sd, "weather": c}); R = 0.0
                while True:
                    o, r, te, tr, i = val_env.step(model.predict(o, deterministic=True)[0]); R += r
                    if te or tr: break
                succ.append(i["outcome"] == "success"); rew.append(R)
        return float(np.mean(succ)), float(np.mean(rew))
    best = (-1.0, -1e18)
    while done < a.steps:
        model.learn(chunk, reset_num_timesteps=False, callback=warm); done += chunk
        eps = [x for x in model.ep_info_buffer]
        rec = {"steps": done, "t_min": (time.time() - t0) / 60, "ep_rew": float(sum(e["r"] for e in eps) / max(1, len(eps))),
               "success": float(sum(e.get("outcome") == "success" for e in eps) / max(1, len(eps)))}
        vs, vr = validate(); rec.update(val_success=vs, val_reward=vr)
        log.append(rec); print(json.dumps(rec), flush=True)
        model.save(out / "final.zip"); model.save(out / f"ckpt_{done}.zip")
        if (vs, vr) > best:
            best = (vs, vr); model.save(out / "best.zip"); (out / "best.json").write_text(json.dumps(rec))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    venv.close()


if __name__ == "__main__":
    main()
