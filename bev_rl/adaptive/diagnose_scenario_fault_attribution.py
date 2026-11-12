"""Fault attribution for a failing scenario: who moves into whom? For the last K steps before the episode ends, log the ego's lane index and
lateral offset in its lane, and for the nearest other vehicle its lane index, lateral offset in ITS lane, heading error to its lane, speed and
relative position. Teacher = stock MetaDrive IDM (lane change off by default here, as in the other diagnostics; --lane-change to enable).
    python -m bev_rl.adaptive.diagnose_scenario_fault_attribution --seeds 1039 1040 --density 0.05 --last 24
"""
import argparse, collections
import numpy as np


def lane_info(o):
    try:
        lane = o.lane; lg, lt = lane.local_coordinates(o.position)
        return tuple(lane.index) if hasattr(lane, "index") else None, round(float(lt), 2), round(float(o.heading_theta - lane.heading_theta_at(lg)), 3)
    except Exception:
        return None, None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--density", type=float, default=0.05)
    ap.add_argument("--last", type=int, default=24); ap.add_argument("--lane-change", action="store_true")
    a = ap.parse_args()
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.env import AdaptiveEnv
    env = AdaptiveEnv(a.seeds, [("clear", 0.0)], weather_input="zeroed", horizon=800, traffic_density=a.density)
    for s in a.seeds:
        env.reset(options={"scenario_seed": s, "weather": ("clear", 0.0)})
        ag = env.md.agent; pol = IDMPolicy(ag, env.md.engine.global_seed); pol.enable_lane_change = a.lane_change
        hist = collections.deque(maxlen=a.last)
        while True:
            act = np.clip(np.asarray(pol.act(), np.float32)[:2], -1, 1)
            th = float(ag.heading_theta); fw = np.array([np.cos(th), np.sin(th)]); rt = np.array([np.sin(th), -np.cos(th)])
            near = None
            for o in env.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
                if o is ag: continue
                d = np.asarray(o.position, float) - np.asarray(ag.position, float)
                if near is None or np.hypot(*d) < near[0]: near = (float(np.hypot(*d)), o, round(float(d @ rt), 2), round(float(d @ fw), 2))
            e_l = lane_info(ag); o_l = lane_info(near[1]) if near else (None,) * 3
            hist.append((env.t, round(float(ag.speed), 1), e_l, round(near[0], 1) if near else None, near[2:] if near else None, round(float(near[1].speed), 1) if near else None, o_l, [round(float(v), 2) for v in act]))
            _, _, te, tr, info = env.step(act)
            if te or tr: break
        print(f"\nseed {s}: {info['outcome']} at step {env.t}, route {info['route_completion']:.2f}")
        print("  t | ego_speed | ego(lane,lat,hdg_err) | dist | other(x_right,y_fwd) | other_speed | other(lane,lat,hdg_err) | action")
        for h in list(hist)[::3] + [hist[-1]]: print("  ", h)

if __name__ == "__main__":
    main()
