"""Gymnasium wrapper: MetaDrive + the existing perception->BEV pipeline.

Observation (actor-visible, nothing else):
    bev    (C, 64, 64) float32  : camera-derived BEV (C=7 for BEV v2; 1 for the legacy v1 BEV)
    weather (4,) float32        : EXPLICIT weather conditioning [onehot(clear, fog, rain), intensity] (supplied, not detected);
                                  all zeros when ``weather_input="zeroed"`` (ablation: same network, no information)
    scalar (6,)        float32  : [speed, route fraction, lateral offset, heading error,
                                   curvature ahead, distance to goal], z-scored with the
                                   BC checkpoint's statistics (so BC and PPO see identical inputs)
Action: Box(-1, 1, (2,)) = [steering, throttle(+)/brake(-)], MetaDrive's native contract.

Weather is sampled per episode from ``weather_conditions`` (or ``options["weather"]``) and applied to the camera
frames by the perception module with a seeded generator. Zero NPC traffic. Simulator truth (lane, route progress, speed) is read
only for the reward and metrics, never returned in the observation.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import gymnasium as gym
import numpy as np

from bev_rl.energy import EnergyParams, step_work_j
from bev_rl.reward import RewardConfig, step_reward
from bev_rl.telemetry import Telemetry, read_telemetry, vehicle_mass_kg


class BEVDrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario_seeds: Sequence[int], scalar_mean, scalar_std, perception: Callable,
                 reward_cfg: RewardConfig | None = None, horizon: int = 800, sample_seed: int = 0,
                 sequential: bool = False, metadrive_env=None, bev_channels: int = 1,
                 weather_conditions: Sequence[tuple] = (), weather_input: str = "explicit"):
        super().__init__()
        self.seeds = list(scenario_seeds)
        self.mean = np.asarray(scalar_mean, dtype=np.float32)
        self.std = np.asarray(scalar_std, dtype=np.float32)
        self.perception = perception          # (md_env, info) -> (bev(64,64), scalar(6,))
        self.cfg = reward_cfg or RewardConfig()
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(sample_seed)
        self.sequential = sequential
        self._next_index = 0
        self.md = metadrive_env
        self.weather_conditions = [tuple(c) for c in weather_conditions]
        if weather_input not in ("explicit", "zeroed"):
            raise ValueError("weather_input must be 'explicit' or 'zeroed'")
        self.weather_input = weather_input
        self.weather = ("clear", 0.0)
        spaces = {"bev": gym.spaces.Box(-1.0, 1.0, (bev_channels, 64, 64), np.float32),
                  "scalar": gym.spaces.Box(-np.inf, np.inf, (6,), np.float32)}
        if self.weather_conditions:
            spaces["weather"] = gym.spaces.Box(0.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.energy_params: EnergyParams | None = None
        self._last_obs = None
        self.bev_hashes: dict = {}      # sha1(BEV bytes) -> frame count, over this env's lifetime

    # ------------------------------------------------------------------ helpers
    def _obs(self, info):
        bev, scalar = self.perception(self.md, info)
        digest = hashlib.sha1(np.ascontiguousarray(bev).tobytes()).hexdigest()[:12]
        self.bev_hashes[digest] = self.bev_hashes.get(digest, 0) + 1
        obs = {"bev": (bev if bev.ndim == 3 else bev[None]).astype(np.float32),
               "scalar": ((scalar - self.mean) / self.std).astype(np.float32)}
        if self.weather_conditions:
            from bev_state.weather import weather_vector
            obs["weather"] = (weather_vector(*self.weather) if self.weather_input == "explicit"
                              else np.zeros(4, dtype=np.float32))
        self._last_obs = obs
        return obs, scalar

    def _pick_seed(self) -> int:
        if self.sequential:
            seed = self.seeds[self._next_index % len(self.seeds)]
            self._next_index += 1
            return seed
        return int(self.rng.choice(self.seeds))

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=None)
        scenario = int((options or {}).get("scenario_seed", self._pick_seed()))
        if self.weather_conditions:
            self.weather = tuple((options or {}).get("weather", self.weather_conditions[int(self.rng.integers(len(self.weather_conditions)))]))
        _, info = self.md.reset(seed=scenario)
        hook = getattr(self.perception, "on_reset", None)
        if hook is not None:
            hook(self.md)
        if hasattr(self.perception, "set_weather"):
            self.perception.set_weather(*self.weather)
        if self.energy_params is None:
            self.energy_params = EnergyParams(mass_kg=vehicle_mass_kg(self.md))
        self.tel: Telemetry = read_telemetry(self.md)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_accel = 0.0
        self.t = 0
        self.energy_j = 0.0
        self.odometer_m = 0.0
        self.scenario_seed = scenario
        obs, _ = self._obs(info)
        return obs, {"scenario_seed": scenario}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dt = self.cfg.dt
        _, _, terminated, truncated, info = self.md.step(action)
        self.t += 1
        prev = self.tel
        tel = read_telemetry(self.md)
        accel = (tel.speed_mps - prev.speed_mps) / dt
        jerk = (accel - self.prev_accel) / dt
        if info.get("arrive_dest"):
            outcome = "success"
        elif info.get("crash_vehicle") or info.get("crash_object") or info.get("crash_building") or info.get("crash_human"):
            outcome = "crash"
        elif info.get("out_of_road"):
            outcome = "off_road"
        else:
            outcome = None
        reward, comps = step_reward(
            self.cfg, ds_m=tel.progress_m - prev.progress_m, speed_mps=tel.speed_mps, lateral_m=tel.lateral_m,
            heading_err_rad=tel.heading_err_rad, action=action, prev_action=self.prev_action,
            jerk_mps3=jerk, outcome=outcome)
        work = step_work_j(prev.speed_mps, tel.speed_mps, dt, self.energy_params)
        self.energy_j += max(0.0, work)
        self.odometer_m += 0.5 * (prev.speed_mps + tel.speed_mps) * dt

        done = bool(terminated)
        time_limit = bool(truncated) or (self.t >= self.horizon and not done)
        if done and not time_limit:
            obs = self._last_obs                       # terminal obs unused by PPO; skip costly perception
        else:
            obs, _ = self._obs(info)
        info_out = {
            "outcome": outcome or ("timeout" if time_limit else None),
            "speed_mps": tel.speed_mps, "lateral_m": tel.lateral_m, "heading_err_rad": tel.heading_err_rad,
            "accel_mps2": accel, "jerk_mps3": jerk, "progress_m": tel.progress_m,
            "route_completion": tel.route_completion, "route_length_m": tel.route_length_m,
            "energy_pos_j": self.energy_j, "odometer_m": self.odometer_m,
            "n_unique_bev": len(self.bev_hashes), "bev_frames": sum(self.bev_hashes.values()),
            "reward_components": comps, "scenario_seed": self.scenario_seed,
            "weather_kind": self.weather[0], "weather_level": float(self.weather[1]),
        }
        self.tel, self.prev_action, self.prev_accel = tel, action, accel
        return obs, reward, done, time_limit and not done, info_out

    def close(self):
        if self.md is not None:
            self.md.close()
            self.md = None
