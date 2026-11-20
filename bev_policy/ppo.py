"""BC-initialized PPO for weather-randomized BEV driving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F

from bev_policy.model import BEVScalarPolicy
from bev_policy.runtime import make_env
from bev_state import BEVStateAssembler
from manual_drive_stereo_yolo_weather import CAMERA_RIGS, frame_from_sensor
from bev_policy.perception import run_perception
from bev_rl.energy import EnergyParams, step_work_j
from bev_rl.telemetry import vehicle_mass_kg


def nearest_forward_obstacle_m(bev: np.ndarray) -> float | None:
    """Estimate front obstacle range from BEV occupancy only (no NPC coordinates)."""
    grid = np.asarray(bev)
    if grid.shape != (64, 64):
        return None
    # Six cells either side of route center span about 3.75 m, enough for a
    # normal lane and car footprint while avoiding adjacent-lane detections.
    for row in range(55, -1, -1):
        if np.any(grid[row, 26:39] >= 0.98):
            return max(0.0, (56 - row) * (20.0 / 64.0))
    return None


class BCAnchoredPPO(PPO):
    """PPO with a behavior-cloning loss that anchors actions to the BC policy."""

    def set_bc_anchor(self, reference: BEVScalarPolicy, loss_coef: float) -> None:
        if loss_coef <= 0.0:
            raise ValueError("BC anchor loss coefficient must be positive.")
        self._bc_reference = reference.to(self.device).eval()
        self._bc_loss_coef = float(loss_coef)
        for parameter in self._bc_reference.parameters():
            parameter.requires_grad_(False)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_bc_reference"]

    def _bc_anchor_loss(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            reference_actions = self._bc_reference(
                observations["bev"], observations["scalar"]
            )
        current_actions = self.policy.get_distribution(observations).get_actions(
            deterministic=True
        )
        return F.mse_loss(current_actions, reference_actions)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses, value_losses = [], [], []
        bc_losses, clip_fractions = [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range),
                ).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = torch.mean(log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                bc_loss = self._bc_anchor_loss(rollout_data.observations)
                bc_losses.append(bc_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self._bc_loss_coef * bc_loss
                )

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching "
                            f"max KL: {approx_kl_div:.4f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/bc_anchor_loss", np.mean(bc_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


class BEVFusionFeatures(BaseFeaturesExtractor):
    """SB3 feature extractor matching the encoders used by BEVScalarPolicy."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 160,
        bev_pool_size: int = 1,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels = int(observation_space["bev"].shape[0])
        scalar_dim = int(observation_space["scalar"].shape[0])
        if bev_pool_size < 1:
            raise ValueError("bev_pool_size must be positive.")
        self.bev_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, 5, 2, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 96, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((bev_pool_size, bev_pool_size)), torch.nn.Flatten(),
            torch.nn.Linear(96 * bev_pool_size * bev_pool_size, 128), torch.nn.ReLU(),
        )
        self.scalar_encoder = torch.nn.Sequential(
            torch.nn.Linear(scalar_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (self.bev_encoder(observations["bev"]),
             self.scalar_encoder(observations["scalar"])),
            dim=1,
        )


class BEVDrivingEnv(gym.Env):
    """Gymnasium adapter that exposes only camera-derived BEV and scalar state."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        yolo_model: Any,
        scalar_mean: np.ndarray,
        scalar_std: np.ndarray,
        *,
        weather: str = "mixed",
        weather_level: float = 0.8,
        random_weather: bool = True,
        traffic_density: float = 0.0,
        max_steps: int = 500,
        render: bool = False,
        seed: int = 0,
        adaptive_speed_penalty: float = 0.05,
        collision_penalty: float = 30.0,
        energy_reward_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.yolo_model = yolo_model
        self.scalar_mean = np.asarray(scalar_mean, dtype=np.float32)
        self.scalar_std = np.maximum(np.asarray(scalar_std, dtype=np.float32), 1e-3)
        self.weather = weather
        self.weather_level = float(weather_level)
        self.random_weather = random_weather
        self.max_steps = int(max_steps)
        self.adaptive_speed_penalty = float(adaptive_speed_penalty)
        self.collision_penalty = float(collision_penalty)
        self.energy_reward_weight = float(energy_reward_weight)
        self.simulator = make_env(
            use_idm=False, render=render, traffic_density=traffic_density, horizon=max_steps
        )
        self.assembler = BEVStateAssembler()
        self.observation_space = spaces.Dict({
            "bev": spaces.Box(-1.0, 1.0, shape=(1, 64, 64), dtype=np.float32),
            "scalar": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.steps = 0
        self.episode_weather = "clear"
        self.episode_level = 0.0
        self.visibility_horizon: float | None = None
        self.weather_rng = np.random.default_rng(seed)
        self.current_state = None
        self.energy_params: EnergyParams | None = None
        self.energy_dt_s: float | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        simulator_seed = (
            int(seed) if seed is not None else int(self.np_random.integers(10, 10010))
        )
        _, info = self.simulator.reset(seed=simulator_seed)
        self.energy_params = EnergyParams(mass_kg=vehicle_mass_kg(self.simulator))
        if "physics_world_step_size" not in self.simulator.config or "decision_repeat" not in self.simulator.config:
            raise RuntimeError("MetaDrive config must provide physics_world_step_size and decision_repeat for energy.")
        self.energy_dt_s = (float(self.simulator.config["physics_world_step_size"])
                            * int(self.simulator.config["decision_repeat"]))
        self.weather_rng = np.random.default_rng(simulator_seed + 20261112)
        if self.random_weather:
            self.episode_weather = str(self.weather_rng.choice(("clear", "fog", "rain")))
            self.episode_level = float(self.weather_rng.uniform(0.0, self.weather_level))
        else:
            self.episode_weather = self.weather
            self.episode_level = self.weather_level
        self.steps = 0
        self.assembler.mount_if_needed(self.simulator)
        observation = self._observation(info)
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        previous_bev = self.current_state.bev_grid if self.current_state is not None else None
        previous_speed = float(self.current_state.scalar_state[0]) if self.current_state is not None else 0.0
        nearest_obstacle = nearest_forward_obstacle_m(previous_bev) if previous_bev is not None else None
        _, reward, terminated, truncated, info = self.simulator.step(action)
        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True
        next_observation = self._observation(info)
        speed_mps = float(self.current_state.scalar_state[0])
        horizon_m = float(self.visibility_horizon) if self.visibility_horizon is not None else 17.5
        # Smooth speed envelope: shorter visible horizon means a lower target.
        weather_speed_limit = float(np.clip(3.0 + 2.5 * np.sqrt(max(horizon_m - 1.0, 0.0)), 4.0, 12.0))
        target_speed = weather_speed_limit
        if nearest_obstacle is not None:
            obstacle_speed_limit = float(np.sqrt(6.0 * max(nearest_obstacle - 3.0, 0.0)))
            target_speed = min(target_speed, obstacle_speed_limit)
        overspeed = max(0.0, speed_mps - target_speed)
        reward -= self.adaptive_speed_penalty * overspeed**2
        if self.energy_params is None or self.energy_dt_s is None:
            raise RuntimeError("Energy parameters must be initialized by reset() before step().")
        step_energy_wh = max(0.0, step_work_j(previous_speed, speed_mps,
                                               self.energy_dt_s, self.energy_params)) / 3600.0
        reward -= self.energy_reward_weight * step_energy_wh
        if info.get("arrive_dest"):
            reward += 10.0
        if info.get("crash_vehicle") or info.get("crash_object"):
            reward -= self.collision_penalty
        if info.get("out_of_road"):
            reward -= self.collision_penalty
        info["bev_nearest_obstacle_m"] = nearest_obstacle
        info["adaptive_speed_limit_mps"] = target_speed
        info["adaptive_speed_penalty"] = self.adaptive_speed_penalty * overspeed**2
        info["estimated_step_energy_wh"] = step_energy_wh
        info["energy_reward_penalty"] = self.energy_reward_weight * step_energy_wh
        return next_observation, float(reward), bool(terminated), bool(truncated), info

    def _observation(self, info: dict[str, Any]) -> dict[str, np.ndarray]:
        frames = {name: frame_from_sensor(self.simulator, name) for name in CAMERA_RIGS}
        conditioned, detections, self.visibility_horizon, front_depth = run_perception(
            frames, self.yolo_model, self.episode_weather, self.episode_level,
            self.weather_rng, stereo_matcher=self.assembler.matcher,
            return_depth=True,
        )
        state = self.assembler.assemble(
            env=self.simulator, info=info, detections_by_camera=detections,
            frames=conditioned, visibility_range_m=self.visibility_horizon,
            front_depth_map=front_depth,
        )
        self.current_state = state
        return {
            "bev": state.bev_grid[None].astype(np.float32, copy=False),
            "scalar": ((state.scalar_state - self.scalar_mean) / self.scalar_std).astype(np.float32),
        }

    def render(self):
        return None

    def close(self) -> None:
        self.simulator.close()


def build_bc_initialized_ppo(
    env: gym.Env,
    bc_checkpoint: Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    n_steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    n_epochs: int = 2,
    target_kl: float = 0.005,
    bc_loss_coef: float = 5.0,
) -> PPO:
    """Create behavior-anchored PPO initialized from the BC policy."""

    saved = torch.load(bc_checkpoint, map_location="cpu", weights_only=True)
    if int(saved["in_channels"]) != 1:
        raise ValueError("PPO currently expects a single-channel 64x64 BEV checkpoint.")
    bev_pool_size = int(saved.get("bev_pool_size", 1))
    model = BCAnchoredPPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=min(batch_size, n_steps),
        n_epochs=n_epochs,
        target_kl=target_kl,
        use_sde=True,
        sde_sample_freq=4,
        seed=seed,
        device=device,
        policy_kwargs={
            "features_extractor_class": BEVFusionFeatures,
            "features_extractor_kwargs": {
                "features_dim": 160,
                "bev_pool_size": bev_pool_size,
            },
            "net_arch": {"pi": [128], "vf": [128]},
            "activation_fn": nn.ReLU,
            "squash_output": True,
        },
        verbose=1,
    )
    bc_policy = BEVScalarPolicy(
        in_channels=1, bev_pool_size=bev_pool_size
    )
    bc_policy.load_state_dict(saved["model_state"])
    model.policy.features_extractor.bev_encoder.load_state_dict(bc_policy.bev_encoder.state_dict())
    model.policy.features_extractor.scalar_encoder.load_state_dict(bc_policy.scalar_encoder.state_dict())
    model.policy.mlp_extractor.policy_net[0].load_state_dict(bc_policy.action_head[0].state_dict())
    model.policy.action_net.load_state_dict(bc_policy.action_head[2].state_dict())
    with torch.no_grad():
        model.policy.log_std.fill_(-1.5)
    model.set_bc_anchor(bc_policy, bc_loss_coef)
    return model
