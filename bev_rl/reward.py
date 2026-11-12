"""Step reward for clear-weather, zero-traffic route following.

Structure follows the PDF's Eq. (26) terms that apply without traffic or an energy
term (the energy-aware reward is a later stage). Weights are re-scaled so that
cruising at ``v_ref`` on the lane centre earns about +1 per step; they are NOT the
PDF's CARLA weights, which assumed 0.04 s steps and a different progress unit.

All inputs are privileged *training signals* (lane geometry, route progress). The
reward is never part of the actor's observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardConfig:
    dt: float = 0.1
    v_ref_mps: float = 10.0          # progress normaliser: +1/step when advancing at v_ref
    v_cap_mps: float = 12.0          # PDF target speed; penalise only above this
    w_progress: float = 1.0
    w_lateral: float = 0.5           # * d^2   (d in metres)
    w_heading: float = 5.0           # * psi^2 (psi in rad)
    w_action_delta: float = 0.1      # * ||a_t - a_{t-1}||^2   (control smoothness)
    w_jerk: float = 0.002            # * (jerk in m/s^3)^2
    w_brake: float = 0.2             # * brake command in [0, 1]  (unnecessary braking)
    w_overspeed: float = 0.5         # * ((v - v_cap)+ / overspeed_scale)^2
    overspeed_scale_mps: float = 2.0
    still_penalty: float = 0.2       # if v < still_speed (anti-stationary)
    still_speed_mps: float = 0.5
    success_bonus: float = 20.0
    crash_penalty: float = -10.0
    offroad_penalty: float = -10.0
    progress_clip_steps: float = 3.0  # clip |ds| to this many reference steps (guards logging glitches)

    def to_dict(self) -> dict:
        return asdict(self)


def step_reward(cfg: RewardConfig, *, ds_m: float, speed_mps: float, lateral_m: float, heading_err_rad: float,
                action, prev_action, jerk_mps3: float, outcome: str | None) -> tuple[float, dict]:
    """Return (reward, components). ``outcome`` in {None,'success','crash','off_road'}."""
    limit = cfg.progress_clip_steps * cfg.v_ref_mps * cfg.dt
    ds = min(max(ds_m, -limit), limit)
    ds = min(ds, cfg.v_cap_mps * cfg.dt)      # no progress credit above the target speed (audit v2: speeding paid)
    comps = {
        "progress": cfg.w_progress * ds / (cfg.v_ref_mps * cfg.dt),
        "lateral": -cfg.w_lateral * lateral_m ** 2,
        "heading": -cfg.w_heading * heading_err_rad ** 2,
        "action_delta": -cfg.w_action_delta * float(sum((float(a) - float(b)) ** 2 for a, b in zip(action, prev_action))),
        "jerk": -cfg.w_jerk * jerk_mps3 ** 2,
        "brake": -cfg.w_brake * max(0.0, -float(action[1])),
        "overspeed": -cfg.w_overspeed * (max(0.0, speed_mps - cfg.v_cap_mps) / cfg.overspeed_scale_mps) ** 2,
        "still": -cfg.still_penalty if speed_mps < cfg.still_speed_mps else 0.0,
        "terminal": {"success": cfg.success_bonus, "crash": cfg.crash_penalty,
                     "off_road": cfg.offroad_penalty}.get(outcome, 0.0),
    }
    return float(sum(comps.values())), comps
