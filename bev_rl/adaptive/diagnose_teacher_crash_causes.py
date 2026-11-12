"""Diagnostic: for seeds where the IDM teacher crashes in traffic, log WHICH crash flag fired, ego speed, and the nearest other object
(relative position in ego frame, its speed, whether it is a vehicle) in the 3 steps before the crash.
    python -m bev_rl.adaptive.diagnose_teacher_crash_causes --seeds 3101 3105 3106 3110 3111 3114 --density 0.2
"""
import argparse, collections
import numpy as np

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--density", type=float, default=0.2); ap.add_argument("--safe", action="store_true")
    a = ap.parse_args()
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.env import AdaptiveEnv
    env = AdaptiveEnv(a.seeds, [("clear", 0.0)], weather_input="zeroed", horizon=800, traffic_density=a.density)
    for s in a.seeds:
        env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
        pol = IDMPolicy(env.md.agent, env.md.engine.global_seed); pol.enable_lane_change = False
        if a.safe:
            from bev_rl.adaptive.safe_teacher import SafeTeacher
            st = SafeTeacher(pol)
        hist = collections.deque(maxlen=4)
        while True:
            act = st.act(env) if a.safe else np.clip(np.asarray(pol.act(), np.float32)[:2], -1, 1)
            ag = env.md.agent; th = float(ag.heading_theta); fw = np.array([np.cos(th), np.sin(th)]); rt = np.array([np.sin(th), -np.cos(th)])
            near = None
            for o in env.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
                if o is ag: continue
                d = np.asarray(o.position, float) - np.asarray(ag.position, float); y, x = float(d @ fw), float(d @ rt)
                if near is None or np.hypot(x, y) < near[0]: near = (np.hypot(x, y), round(x, 1), round(y, 1), round(float(o.speed), 1), type(o).__name__)
            hist.append((round(float(ag.speed), 1), near, [round(float(v), 2) for v in act]))
            _, _, te, tr, info = env.step(act)
            if te or tr:
                break
        ag = env.md.agent
        flags = {k: bool(getattr(ag, k, False)) for k in ("crash_vehicle", "crash_object", "crash_building", "crash_sidewalk")}
        print(f"seed {s}: outcome={info['outcome']} after {env.t} steps, flags={flags}; last frames (ego_speed, nearest(dist,x_right,y_fwd,speed,type), action):")
        for h in hist: print("   ", h)
    return

if __name__ == "__main__":
    main()
