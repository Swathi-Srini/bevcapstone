"""Reward sanity audit: scripted behaviours must rank sensibly under the step reward.

Behaviours share the BC steering (computed with the speed input fixed at 8 m/s for crawl/fast) so they stay on the road; only the longitudinal command differs:
  bc          : BC policy as is
  stationary  : no throttle, no brake
  crawl       : P-controller to 2 m/s
  fast        : P-controller to 18 m/s (above the 12 m/s cap)
Expected ordering of total return over the same horizon: bc > crawl > stationary, and fast must not beat bc.
"""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seeds", default="2000-2002"); p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--out", type=Path, default=Path("bev_rl/results/reward_audit.json")); a = p.parse_args()
    from bev_rl.evaluate import bc_act_fn, parse_seeds, run_episode
    from bev_rl.factory import make_driving_env
    from bev_rl.policy import load_bc_checkpoint
    ck = load_bc_checkpoint("bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt")
    seeds = parse_seeds(a.seeds)
    env = make_driving_env(seeds, ck["scalar_mean"], ck["scalar_std"], horizon=a.horizon, sequential=True, cached_bev=True)
    bc = bc_act_fn(ck)
    def with_speed(target):
        def act(obs):
            o = {"bev": obs["bev"], "scalar": obs["scalar"].copy()}
            o["scalar"][0] = (8.0 - env.mean[0]) / env.std[0]   # BC steers from its familiar ~8 m/s regime (v1 audit was confounded)
            steer = bc(o)[0]
            v = obs["scalar"][0] * env.std[0] + env.mean[0]
            return np.array([steer, np.clip(0.5 * (target - v), -1, 1)], dtype=np.float32)
        return act
    behaviours = {"bc": bc, "stationary": lambda o: np.array([0.0, 0.0], np.float32),
                  "crawl": with_speed(2.0), "fast": with_speed(18.0)}
    out = {}
    try:
        for name, act in behaviours.items():
            rows = [run_episode(env, act, s)[0] for s in seeds]
            out[name] = {k: float(np.mean([r[k] for r in rows])) for k in
                         ("episode_reward", "route_completion", "success", "mean_speed_mps", "steps", "rms_lateral_m", "wall_s")}
            out[name]["outcomes"] = [r["outcome"] for r in rows]
            print(name, out[name], flush=True)
    finally:
        env.close()
    out["perception_audit"] = {"audits": env.perception.audits, "mismatches": env.perception.audit_mismatches}
    out["ordering_ok"] = bool(out["bc"]["episode_reward"] > out["crawl"]["episode_reward"] > out["stationary"]["episode_reward"]
                              and out["fast"]["episode_reward"] < out["bc"]["episode_reward"])
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
