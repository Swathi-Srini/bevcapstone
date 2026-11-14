"""How hard can the simulated ego car actually brake? (Needed before the stopping-distance rule can assume a deceleration b.)

Drive straight with the stock IDM to a target speed, then command full brake (throttle/brake action = -1) and record speed and distance every 0.1 s until standstill.
Reports per start speed: stopping distance, mean deceleration v0^2/(2 d), peak deceleration, and time to stop. Simulator only, no camera. Weather has no effect on vehicle dynamics in this simulator
(friction is constant), so one result applies to every weather condition.

    python -m bev_rl.adaptive.brake_check
"""
import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main():
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.caution import CautionEnv
    env = CautionEnv([3100], [("clear", 0.0)], dropout=False, traffic_density=0.0, horizon=600, sample_seed=7)
    for seed in (3100, 3101, 3102):
        for v0 in (4.0, 6.0, 8.3):
            env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); idm = IDMPolicy(env.md.agent, env.md.engine.global_seed); idm.NORMAL_SPEED = 40.0; idm.target_speed = 40.0
            for _ in range(400):                                                       # accelerate to v0 with the IDM's own steering
                a = np.clip(np.asarray(idm.act(), np.float32)[:2], -1, 1)
                if float(env.md.agent.speed) >= v0: break
                env.step(a)
            if float(env.md.agent.speed) < v0 - 0.3: print(f"seed {seed} v0 {v0}: could not reach speed"); continue
            v_start = float(env.md.agent.speed); p0 = np.asarray(env.md.agent.position, float).copy(); speeds = [v_start]; ok = True
            for _ in range(200):
                a = np.clip(np.asarray(idm.act(), np.float32)[:2], -1, 1); a[1] = -1.0
                _, _, te, tr, info = env.step(a); speeds.append(float(env.md.agent.speed))
                if te or tr: ok = False; break
                if speeds[-1] < 0.05: break
            d = float(np.linalg.norm(np.asarray(env.md.agent.position, float) - p0)); s = np.array(speeds); dec = -np.diff(s) / 0.1
            print(f"seed {seed} start {v_start:5.2f} m/s: stop distance {d:6.2f} m, mean decel {v_start**2 / (2 * d):5.2f} m/s^2, peak decel {dec.max():5.2f}, time to stop {0.1 * (len(s) - 1):4.1f} s{'' if ok else '  (episode ended early)'}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
