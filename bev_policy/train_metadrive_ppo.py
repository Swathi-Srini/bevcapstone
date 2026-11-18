"""Train and evaluate a fast MetaDrive PPO driving policy.

This is a simulator-state RL demonstration, separate from the camera/BEV
weather-adaptation experiment in ``train_ppo.py``. MetaDrive's 259-D state
observation keeps training practical on CPU and uses its native driving reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

from baseline.bc_metadrive.model import BCPolicy


def make_env(*, traffic_density: float, horizon: int, render: bool, use_idm: bool = False):
    from metadrive import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    config = {
        "use_render": render,
        "manual_control": False,
        "traffic_density": traffic_density,
        "num_scenarios": 10000,
        "start_seed": 10,
        "horizon": horizon,
        "map": 4,
        "random_agent_model": False,
        "random_lane_width": False,
        "random_lane_num": False,
        "out_of_route_done": True,
        "on_continuous_line_done": False,
        "use_lateral_reward": True,
        "success_reward": 20.0,
        "out_of_road_penalty": 10.0,
        "crash_vehicle_penalty": 10.0,
        "crash_object_penalty": 10.0,
        "speed_reward": 0.02,
        "vehicle_config": {
            "show_lidar": True,
            "show_navi_mark": False,
            "show_line_to_navi_mark": False,
        },
    }
    if use_idm:
        config["agent_policy"] = IDMPolicy
    return MetaDriveEnv(config)


def collect_expert_data(*, episodes: int, seed: int, traffic_density: float,
                        horizon: int) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(traffic_density=traffic_density, horizon=horizon, render=False, use_idm=True)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    accepted = 0
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index)
            episode_obs: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            outcome = "max_steps"
            for _ in range(horizon):
                episode_obs.append(np.asarray(obs, dtype=np.float32))
                obs, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
                episode_actions.append(np.clip(
                    np.asarray(env.agent.current_action, dtype=np.float32), -1.0, 1.0
                ))
                if info.get("arrive_dest"):
                    outcome = "success"
                elif info.get("crash_vehicle") or info.get("crash_object"):
                    outcome = "collision"
                elif info.get("out_of_road"):
                    outcome = "off_road"
                if terminated or truncated:
                    break
            if outcome == "success":
                observations.extend(episode_obs)
                actions.extend(episode_actions)
                accepted += 1
            print(f"expert {index + 1}/{episodes}: seed={seed + index}, "
                  f"outcome={outcome}, samples={len(episode_obs)}", flush=True)
    finally:
        env.close()
    if accepted < 2:
        raise RuntimeError(f"Only {accepted} expert episodes reached the destination; need at least two.")
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def collect_dagger_data(model: BCPolicy, *, episodes: int, seed: int,
                        traffic_density: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Query IDM on learner-visited states to reduce covariate shift."""
    from metadrive.policy.idm_policy import IDMPolicy

    env = make_env(traffic_density=traffic_density, horizon=horizon, render=False)
    adapter = ImitationPolicyAdapter(model)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            obs, _ = env.reset(seed=episode_seed)
            expert = IDMPolicy(env.agent, episode_seed)
            outcome = "max_steps"
            count = 0
            try:
                for _ in range(horizon):
                    student_action, _ = adapter.predict(obs)
                    expert_action = np.clip(
                        np.asarray(expert.act(), dtype=np.float32), -1.0, 1.0
                    )
                    observations.append(np.asarray(obs, dtype=np.float32))
                    actions.append(expert_action)
                    obs, _, terminated, truncated, info = env.step(student_action)
                    count += 1
                    if info.get("arrive_dest"):
                        outcome = "success"
                    elif info.get("crash_vehicle") or info.get("crash_object"):
                        outcome = "collision"
                    elif info.get("out_of_road"):
                        outcome = "off_road"
                    if terminated or truncated:
                        break
            finally:
                expert.destroy()
            print(f"DAgger {index + 1}/{episodes}: seed={episode_seed}, "
                  f"learner_outcome={outcome}, labeled_steps={count}", flush=True)
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_imitation_warmstart(observations: np.ndarray, actions: np.ndarray,
                              *, epochs: int, seed: int) -> BCPolicy:
    torch.manual_seed(seed)
    model = BCPolicy()
    dataset = TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for obs, action in loader:
            squared_error = (model(obs) - action).square()
            steering_weights = 1.0 + 4.0 * (action[:, 0].abs() > 0.05).float()
            weights = torch.ones_like(squared_error)
            weights[:, 0] = steering_weights
            loss = (squared_error * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(obs)
        print(f"imitation epoch {epoch + 1}/{epochs}: loss={total / len(dataset):.6f}", flush=True)
    return model.eval()


class ImitationPolicyAdapter:
    def __init__(self, model: BCPolicy, *, apply_tanh: bool = False):
        self.model = model.eval()
        self.apply_tanh = apply_tanh

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        del deterministic
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32).reshape(1, -1)
            action = self.model(tensor).squeeze(0)
            if self.apply_tanh:
                action = torch.tanh(action)
            action = action.cpu().numpy()
        return action, None


def initialise_ppo_actor(policy: Any, imitation: BCPolicy) -> None:
    source = imitation.state_dict()
    layers = policy.mlp_extractor.policy_net
    mapping = {
        "feature_extractor.0.weight": layers[0].weight,
        "feature_extractor.0.bias": layers[0].bias,
        "feature_extractor.2.weight": layers[2].weight,
        "feature_extractor.2.bias": layers[2].bias,
        "policy_head.0.weight": layers[4].weight,
        "policy_head.0.bias": layers[4].bias,
        "policy_head.2.weight": policy.action_net.weight,
        "policy_head.2.bias": policy.action_net.bias,
    }
    with torch.no_grad():
        for name, parameter in mapping.items():
            parameter.copy_(source[name])


def evaluate(model: PPO, *, episodes: int, seed: int, traffic_density: float,
             horizon: int, render: bool) -> list[dict[str, Any]]:
    env = make_env(traffic_density=traffic_density, horizon=horizon, render=render)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            obs, _ = env.reset(seed=episode_seed)
            speeds: list[float] = []
            actions: list[np.ndarray] = []
            outcome = "max_steps"
            for step in range(1, horizon + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append(np.asarray(action, dtype=np.float32))
                speed = getattr(env.agent, "speed", 0.0)
                speeds.append(float(speed() if callable(speed) else speed))
                if info.get("arrive_dest"):
                    outcome = "success"
                elif info.get("crash_vehicle") or info.get("crash_object"):
                    outcome = "collision"
                elif info.get("out_of_road"):
                    outcome = "off_road"
                if terminated or truncated:
                    break
            rows.append({
                "seed": episode_seed,
                "outcome": outcome,
                "steps": step,
                "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                "mean_abs_steering": float(np.mean(np.abs([a[0] for a in actions]))) if actions else 0.0,
                "mean_throttle_brake_command": float(np.mean([a[1] for a in actions])) if actions else 0.0,
            })
            print(f"evaluation {index + 1}/{episodes}: seed={episode_seed}, "
                  f"outcome={outcome}, steps={step}", flush=True)
    finally:
        env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--traffic-density", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--expert-episodes", type=int, default=20)
    parser.add_argument("--imitation-epochs", type=int, default=25)
    parser.add_argument("--dagger-episodes", type=int, default=20)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--output", type=Path,
                        default=Path("bev_policy/checkpoints/metadrive_ppo_light_traffic"))
    parser.add_argument("--imitation-checkpoint", type=Path,
                        default=Path("bev_policy/checkpoints/metadrive_idm_warmstart.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("bev_policy/outputs/metadrive_ppo_light_traffic.json"))
    parser.add_argument("--render-eval", action="store_true")
    args = parser.parse_args()
    if args.timesteps < 1 or args.horizon < 1 or args.eval_episodes < 1:
        parser.error("timesteps, horizon, and eval-episodes must be positive")
    if args.traffic_density < 0:
        parser.error("traffic-density must be non-negative")

    expert_obs, expert_actions = collect_expert_data(
        episodes=args.expert_episodes, seed=300,
        traffic_density=args.traffic_density, horizon=args.horizon,
    )
    imitation = train_imitation_warmstart(
        expert_obs, expert_actions, epochs=args.imitation_epochs, seed=args.seed,
    )
    all_obs, all_actions = expert_obs, expert_actions
    for round_index in range(args.dagger_rounds):
        dagger_obs, dagger_actions = collect_dagger_data(
            imitation, episodes=args.dagger_episodes,
            seed=500 + round_index * args.dagger_episodes,
            traffic_density=args.traffic_density, horizon=args.horizon,
        )
        all_obs = np.concatenate((all_obs, dagger_obs))
        all_actions = np.concatenate((all_actions, dagger_actions))
        imitation = train_imitation_warmstart(
            all_obs, all_actions, epochs=args.imitation_epochs,
            seed=args.seed + round_index + 1,
        )
    args.imitation_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": imitation.state_dict(),
        "observation_dim": int(expert_obs.shape[1]),
        "expert_episodes": args.expert_episodes,
        "accepted_samples": int(len(all_obs)),
        "initial_expert_samples": int(len(expert_obs)),
        "dagger_rounds": args.dagger_rounds,
        "traffic_density": args.traffic_density,
        "expert_seed_start": 300,
        "expert": "MetaDrive IDMPolicy",
    }, args.imitation_checkpoint)
    imitation_rows = evaluate(
        ImitationPolicyAdapter(imitation), episodes=args.eval_episodes,
        seed=args.eval_seed, traffic_density=args.traffic_density,
        horizon=args.horizon, render=False,
    )
    env = make_env(traffic_density=args.traffic_density, horizon=args.horizon, render=False)
    try:
        model = PPO(
            "MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
            learning_rate=1e-5, n_steps=1024, batch_size=256, n_epochs=2,
            gamma=0.99, gae_lambda=0.95, clip_range=0.15, ent_coef=0.001,
            target_kl=0.005, use_sde=True,
            policy_kwargs={"net_arch": [256, 256, 128],
                           "squash_output": True},
        )
        initialise_ppo_actor(model.policy, imitation)
        with torch.no_grad():
            model.policy.log_std.fill_(-1.5)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output))
    finally:
        env.close()

    report_episodes = evaluate(
        model, episodes=args.eval_episodes, seed=args.eval_seed,
        traffic_density=args.traffic_density, horizon=args.horizon,
        render=args.render_eval,
    )
    outcomes = [row["outcome"] for row in report_episodes]
    summary = {
        "episodes": len(report_episodes),
        "success_rate": float(np.mean([x == "success" for x in outcomes])),
        "collision_rate": float(np.mean([x == "collision" for x in outcomes])),
        "off_road_rate": float(np.mean([x == "off_road" for x in outcomes])),
        "mean_speed_mps": float(np.mean([r["mean_speed_mps"] for r in report_episodes])),
    }
    report = {
        "algorithm": "Stable-Baselines3 PPO",
        "observation": "MetaDrive 259-D simulator state",
        "traffic_density": args.traffic_density,
        "training_timesteps": args.timesteps,
        "training_seed": args.seed,
        "evaluation_seed": args.eval_seed,
        "horizon": args.horizon,
        "episodes": report_episodes,
        "summary": summary,
        "imitation_warmstart": {
            "checkpoint": str(args.imitation_checkpoint),
            "samples": int(len(expert_obs)),
            "evaluation_episodes": imitation_rows,
        },
        "limitation": "Simulator-state RL demo; it does not use camera BEV or weather-conditioned perception.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved driving policy: {args.output}.zip")
    print(f"Saved evaluation report: {args.report}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
