"""PPO fine-tuning of the BC-initialised camera-BEV(v2) actor-critic, optionally weather-conditioned (zero traffic).

    python -m bev_rl.train_ppo --seed 0 --total-steps 20000 --out bev_rl/runs/ppo_s0

Everything needed to reproduce a run is written to ``<out>/config.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=20_000)
    p.add_argument("--init", type=Path, default=Path("bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt"))
    p.add_argument("--train-seeds", default="2000-2999")
    p.add_argument("--horizon", type=int, default=800)
    p.add_argument("--n-steps", type=int, default=512)       # PDF Table 14
    p.add_argument("--batch-size", type=int, default=64)     # PDF Table 14
    p.add_argument("--n-epochs", type=int, default=10)       # PDF Table 14
    p.add_argument("--lr", type=float, default=3e-4)         # PDF Table 14 (BC-init fine-tuning may need less)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)    # PDF: 0.01; 0 here to avoid destroying the BC policy
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--log-std-init", type=float, default=-1.6)
    p.add_argument("--critic-warmup-rollouts", type=int, default=2,
                   help="Rollouts during which the actor is frozen and only the critic learns.")
    p.add_argument("--yolo-weights", default="weights/yolov8n.pt")
    p.add_argument("--conditions", default="clear:0",
                   help="Training weather set, e.g. 'clear:0' or 'clear:0,fog:0.3,fog:0.6,rain:0.3,rain:0.6'.")
    p.add_argument("--weather-input", choices=("explicit", "zeroed"), default="explicit",
                   help="'zeroed' = ablation: identical network, weather vector forced to zero.")
    p.add_argument("--cam-scale", type=float, default=0.5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from bev_rl.evaluate import parse_seeds
    from bev_rl.factory import make_v2_env
    from bev_rl.policy import BEVActorCritic, init_from_bc, load_bc_checkpoint
    from bev_rl.reward import RewardConfig

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt = load_bc_checkpoint(args.init)
    reward_cfg = RewardConfig()
    train_seeds = parse_seeds(args.train_seeds)
    conditions = [(c.split(":")[0], float(c.split(":")[1])) for c in args.conditions.split(",")]
    env = make_v2_env(train_seeds, ckpt["scalar_mean"], ckpt["scalar_std"], weather_conditions=conditions,
                      weather_input=args.weather_input, horizon=args.horizon, sample_seed=args.seed, reward_cfg=reward_cfg,
                      cam_scale=args.cam_scale)
    venv = DummyVecEnv([lambda: Monitor(env, str(args.out / "monitor"), info_keywords=("outcome", "route_completion"))])
    venv.seed(args.seed)

    model = PPO(BEVActorCritic, venv, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda, clip_range=args.clip_range,
                ent_coef=args.ent_coef, target_kl=args.target_kl, policy_kwargs={"log_std_init": args.log_std_init},
                seed=args.seed, device="cpu", verbose=1)
    init_from_bc(model.policy, ckpt)
    model.set_logger(configure(str(args.out / "sb3"), ["stdout", "csv"]))

    class Schedule(BaseCallback):
        """Critic warm-up (actor frozen) and a checkpoint after every rollout."""

        def _actor_params(self):
            pol = self.model.policy
            return (list(pol.pi_features_extractor.parameters()) + list(pol.mlp_extractor.policy_net.parameters())
                    + list(pol.action_net.parameters()) + [pol.log_std])

        def _on_training_start(self):
            self.rollouts = 0
            if args.critic_warmup_rollouts > 0:
                for prm in self._actor_params():
                    prm.requires_grad_(False)

        def _on_rollout_end(self):
            self.rollouts += 1
            if self.rollouts == args.critic_warmup_rollouts:
                for prm in self._actor_params():
                    prm.requires_grad_(True)
                print(f"[schedule] critic warm-up finished after {self.rollouts} rollouts; actor unfrozen", flush=True)
            self.model.save(args.out / f"ckpt_{self.num_timesteps}.zip")

        def _on_step(self):
            return True

    config = {"args": {k: str(v) for k, v in vars(args).items()}, "reward": reward_cfg.to_dict(),
              "bc_init_stats": {"scalar_mean": ckpt["scalar_mean"], "scalar_std": ckpt["scalar_std"]},
              "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "git": git_state()}
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    try:
        model.learn(total_timesteps=args.total_steps, callback=Schedule(), progress_bar=False)
        model.save(args.out / "final.zip")
    finally:
        perc = env.perception
        audit = {"timing_s_sum": perc.timing, "unique_bev": len(env.bev_hashes), "bev_frames": sum(env.bev_hashes.values())}
        (args.out / "perception_audit.json").write_text(json.dumps(audit, indent=2))
        print("[perception audit]", audit, flush=True)
        venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
