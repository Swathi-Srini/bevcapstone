"""Reward v3 for weather-adaptive driving in sparse traffic (replaces bev_rl/reward.py for this study; the old file is untouched).

What was wrong with the old reward (measured/read 2026-11-10):
  * it described itself as 'zero-traffic route following': NO term about other vehicles, so a crash was the only traffic signal (rare, sparse, late credit assignment);
  * it PENALISED BRAKING (w_brake) -> directly discouraged the behaviour that perception loss should call for;
  * crash -10 (or -50 in PPO runs) against ~+1/step progress: dominated by progress, and the energy term was a flat per-Wh penalty.
Design principles of v3 (all signals are PRIVILEGED training signals; the policy never sees them):
  1. dense, early safety signal from ground truth:  TTC penalty  w_ttc * (1 - ttc/ttc_ref)^2  for ttc < ttc_ref  (constant-velocity TTC, SafeTeacher.ttc geometry)
                                                    gap penalty  w_gap * (1 - gap/gap_ref)^2  for gap < gap_ref   (bumper gap to the nearest vehicle ahead in the corridor)
     -> when perception is degraded the policy that keeps larger margins earns more, so caution EMERGES from the reward instead of a hand-set speed rule;
  2. terminal: success +50, crash / off-road -100 (about the return forfeited by terminating early, ~0.8/step x ~400 steps = ~320 is NOT matched on purpose: the TTC/gap terms
     carry the dense signal and the terminal term only has to dominate one step of speed-up);
  3. no brake penalty; anti-stationary penalty kept (otherwise 'stop' is a trivial safe policy);
  4. energy: -w_energy * Wh per step (positive traction work), so slowing down has a measured price and a measured benefit;
  5. comfort: action-delta and jerk (as before, small).
Progress credit is capped at v_cap (no reward for speeding).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardV3:
    dt: float = 0.1
    v_ref_mps: float = 10.0
    v_cap_mps: float = 10.0
    w_progress: float = 1.0
    w_lateral: float = 0.5
    w_heading: float = 5.0
    w_ttc: float = 2.0
    ttc_ref_s: float = 3.0
    w_gap: float = 1.0
    gap_ref_m: float = 6.0
    w_action_delta: float = 0.1
    w_jerk: float = 0.002
    w_energy: float = 0.02
    still_penalty: float = 0.2
    still_speed_mps: float = 0.5
    success_bonus: float = 50.0
    crash_penalty: float = -100.0
    offroad_penalty: float = -100.0
    progress_clip_steps: float = 3.0


def step_reward_v3(cfg, *, ds_m, speed_mps, lateral_m, heading_err_rad, action, prev_action, jerk_mps3, ttc_s, gap_m, wh, outcome):
    """-> (reward, components). ttc_s: None if no predicted collision within the horizon; gap_m: bumper gap ahead (large if none); wh: positive traction Wh this step."""
    limit = cfg.progress_clip_steps * cfg.v_ref_mps * cfg.dt
    ds = min(min(max(ds_m, -limit), limit), cfg.v_cap_mps * cfg.dt)
    ttc_pen = 0.0 if ttc_s is None or ttc_s >= cfg.ttc_ref_s else cfg.w_ttc * (1.0 - max(ttc_s, 0.0) / cfg.ttc_ref_s) ** 2
    gap_pen = 0.0 if gap_m >= cfg.gap_ref_m else cfg.w_gap * (1.0 - max(gap_m, 0.0) / cfg.gap_ref_m) ** 2
    comps = {
        "progress": cfg.w_progress * ds / (cfg.v_ref_mps * cfg.dt),
        "lateral": -cfg.w_lateral * lateral_m ** 2,
        "heading": -cfg.w_heading * heading_err_rad ** 2,
        "ttc": -ttc_pen,
        "gap": -gap_pen,
        "action_delta": -cfg.w_action_delta * float(sum((float(a) - float(b)) ** 2 for a, b in zip(action, prev_action))),
        "jerk": -cfg.w_jerk * jerk_mps3 ** 2,
        "energy": -cfg.w_energy * wh,
        "still": -cfg.still_penalty if speed_mps < cfg.still_speed_mps else 0.0,
        "terminal": {"success": cfg.success_bonus, "crash": cfg.crash_penalty, "off_road": cfg.offroad_penalty}.get(outcome, 0.0),
    }
    return float(sum(comps.values())), comps


if __name__ == "__main__":                     # self-test: cruising, closing on a stopped car, crash
    c = RewardV3()
    base = dict(ds_m=0.8, speed_mps=8.0, lateral_m=0.0, heading_err_rad=0.0, action=(0, 0.3), prev_action=(0, 0.3), jerk_mps3=0.0, wh=0.05, outcome=None)
    print("cruise            ", round(step_reward_v3(c, ttc_s=None, gap_m=30.0, **base)[0], 3))
    print("ttc 1.5 s, gap 4 m", round(step_reward_v3(c, ttc_s=1.5, gap_m=4.0, **base)[0], 3))
    print("ttc 0.3 s, gap 1 m", round(step_reward_v3(c, ttc_s=0.3, gap_m=1.0, **base)[0], 3))
    print("crash             ", round(step_reward_v3(c, ttc_s=0.0, gap_m=0.0, **{**base, "outcome": "crash"})[0], 3))
    assert step_reward_v3(c, ttc_s=1.5, gap_m=4.0, **base)[0] < step_reward_v3(c, ttc_s=None, gap_m=30.0, **base)[0]
    print("ok")
