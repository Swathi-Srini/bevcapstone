"""Fast camera-free MetaDrive environment for weather-adaptive, energy-aware driving (privileged teacher).

Observation (same information the camera-BEV student receives besides pixels):
    [speed, route_fraction, lateral, heading_err, curvature_ahead, dist_to_goal] (6-D, scaled) + prev action (2)
    + explicit weather vector [onehot(clear,fog,rain), intensity] (4) (zeroed for the unconditioned ablation).
Physics coupling (DECLARED ASSUMPTION): rain lowers tyre-road friction  mu = mu0 * (1 - RAIN_MU_LOSS * intensity).
Fog changes only perception (it has no dynamics effect here). Zero traffic.
Reward: progress/lateral/heading/smoothness/overspeed terms of bev_rl.reward plus  -lambda_e * (positive traction Wh per step).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from bev_rl.energy import EnergyParams, JOULES_PER_WH, step_work_j
from bev_rl.reward import RewardConfig, step_reward
from bev_rl.telemetry import read_telemetry, vehicle_mass_kg


def weather_vector(kind: str, intensity: float) -> np.ndarray:
    """Policy-side explicit weather input [onehot(clear, fog, rain), intensity] (defined here; no dependency on the weather module)."""
    v = np.zeros(4, dtype=np.float32)
    v[("clear", "fog", "rain").index(kind)] = 1.0
    v[3] = 0.0 if kind == "clear" else float(np.clip(intensity, 0.0, 1.0))
    return v


MU0 = 0.9
RAIN_MU_LOSS = 0.65        # rain 1.0 -> mu = 0.315
SCALE = np.array([10.0, 1.0, 2.0, 0.5, 0.05, 300.0], np.float32)


def mu_for(kind: str, level: float) -> float:
    return MU0 * (1.0 - RAIN_MU_LOSS * level) if kind == "rain" else MU0


def make_md(horizon: int, expert: bool = False, traffic_density: float = 0.0):
    from metadrive import MetaDriveEnv
    extra = {}
    if expert:
        from metadrive.policy.idm_policy import IDMPolicy
        extra = {"agent_policy": IDMPolicy}
    return MetaDriveEnv({**extra, "use_render": False, "image_observation": False, "traffic_density": float(traffic_density), "num_scenarios": 10000,
                         "start_seed": 10, "horizon": horizon, "random_agent_model": False, "random_lane_width": True,
                         "random_lane_num": True, "out_of_route_done": True, "on_continuous_line_done": False,
                         "manual_control": False, "log_level": 50})


class AdaptiveEnv(gym.Env):
    def __init__(self, seeds, conditions, weather_input="explicit", lambda_e=0.0, horizon=800, sample_seed=0,
                 v_cap=12.0, sequential=False, expert=False, crash_pen=10.0, traffic_density=0.0):
        from bev_state import BEVStateAssembler
        self.asm = BEVStateAssembler.__new__(BEVStateAssembler) if False else BEVStateAssembler()
        self.seeds, self.conditions = list(seeds), [tuple(c) for c in conditions]
        self.weather_input, self.lambda_e, self.horizon = weather_input, float(lambda_e), int(horizon)
        self.cfg = RewardConfig(v_cap_mps=v_cap, crash_penalty=-float(crash_pen), offroad_penalty=-float(crash_pen))
        self.rng = np.random.default_rng(sample_seed)
        self.sequential, self._i = sequential, 0
        self.expert = expert
        self.md = make_md(horizon, expert, traffic_density)
        self.observation_space = gym.spaces.Box(-5, 5, (12,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self.energy_params = None

    def _obs(self, info):
        sc = (self.asm.extract_scalar_state(self.md, info) / SCALE).astype(np.float32)
        w = weather_vector(*self.weather) if self.weather_input == "explicit" else np.zeros(4, np.float32)
        return np.clip(np.concatenate([sc, self.prev_action, w]), -5, 5).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        o = options or {}
        if "scenario_seed" in o: sc = int(o["scenario_seed"])
        elif self.sequential: sc = self.seeds[self._i % len(self.seeds)]; self._i += 1
        else: sc = int(self.rng.choice(self.seeds))
        self.weather = tuple(o.get("weather", self.conditions[int(self.rng.integers(len(self.conditions)))]))
        _, info = self.md.reset(seed=sc)
        self.mu = mu_for(*self.weather)
        for w in self.md.agent.system.getWheels():
            w.setFrictionSlip(self.mu)
        if self.energy_params is None:
            self.energy_params = EnergyParams(mass_kg=vehicle_mass_kg(self.md))
        self.tel = read_telemetry(self.md)
        self.prev_action = np.zeros(2, np.float32); self.prev_accel = 0.0
        self.t = 0; self.energy_j = 0.0; self.odometer = 0.0; self.scenario = sc
        self.speeds, self.lat = [], []
        return self._obs(info), {}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        dt = self.cfg.dt
        _, _, term, trunc, info = self.md.step(a)
        if self.expert:
            a = np.clip(np.asarray(self.md.agent.current_action, np.float32), -1, 1)
        self.t += 1
        prev, tel = self.tel, read_telemetry(self.md)
        accel = (tel.speed_mps - prev.speed_mps) / dt; jerk = (accel - self.prev_accel) / dt
        if info.get("arrive_dest"): outcome = "success"
        elif any(info.get(k) for k in ("crash_vehicle", "crash_object", "crash_building", "crash_human")): outcome = "crash"
        elif info.get("out_of_road"): outcome = "off_road"
        else: outcome = None
        r, comps = step_reward(self.cfg, ds_m=tel.progress_m - prev.progress_m, speed_mps=tel.speed_mps, lateral_m=tel.lateral_m,
                               heading_err_rad=tel.heading_err_rad, action=a, prev_action=self.prev_action, jerk_mps3=jerk, outcome=outcome)
        work = max(0.0, step_work_j(prev.speed_mps, tel.speed_mps, dt, self.energy_params))
        wh = work / JOULES_PER_WH
        r -= self.lambda_e * wh
        self.energy_j += work; self.odometer += 0.5 * (prev.speed_mps + tel.speed_mps) * dt
        self.speeds.append(tel.speed_mps); self.lat.append(abs(tel.lateral_m))
        done = bool(term); tl = bool(trunc) or (self.t >= self.horizon and not done)
        self.tel, self.prev_action, self.prev_accel = tel, a, accel
        out = {"outcome": outcome or ("timeout" if tl else None), "route_completion": tel.route_completion,
               "energy_wh": self.energy_j / JOULES_PER_WH, "distance_m": self.odometer, "mean_speed": float(np.mean(self.speeds)),
               "mean_abs_lat": float(np.mean(self.lat)), "max_abs_lat": float(np.max(self.lat)), "steps": self.t,
               "weather_kind": self.weather[0], "weather_level": float(self.weather[1]), "mu": self.mu, "scenario": self.scenario}
        return self._obs(info), float(r), done, tl and not done, out

    def truth_objects(self):
        """Ideal-detector stand-in: simulator vehicles inside the BEV window as GitHub ObjectFootprints (ego frame)."""
        from bev_state.assembler import ObjectFootprint
        agent = self.md.agent
        ego, th = np.asarray(agent.position, float), float(agent.heading_theta)
        right, fwd = np.array([np.sin(th), -np.cos(th)]), np.array([np.cos(th), np.sin(th)])
        out = []
        for o in self.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
            if o is agent:
                continue
            d = np.asarray(o.position, float) - ego
            x, y = float(d @ right), float(d @ fwd)
            if -10.0 <= x <= 10.0 and -2.5 <= y <= 17.5:
                out.append(ObjectFootprint("car", x, y, float(getattr(o, "WIDTH", 1.85)), float(getattr(o, "LENGTH", 4.5)), 1.0, "truth", "truth"))
        return out

    overlay = False        # set True to fill GitHub's reserved route(0.5)/boundary(0.8) values from the HD map (approved 2026-11-04)

    def bev(self, visibility_range_m=None, objects=None):
        """64x64 GitHub BEV (BEVStateAssembler.build_bev_grid, unmodified). Default objects = ideal detections."""
        g = self.asm.build_bev_grid(self.truth_objects() if objects is None else objects, visibility_range_m)
        if self.overlay:
            from bev_rl.adaptive.spec_bev import overlay_github_bev
            g = overlay_github_bev(g, self.md)
        return g

    def close(self):
        if self.md is not None: self.md.close(); self.md = None
