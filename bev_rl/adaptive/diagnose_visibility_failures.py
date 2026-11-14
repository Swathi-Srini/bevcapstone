"""Replay individual FAILED episodes of the visibility-teacher prototype and log what happened in the last steps. Simulator only. Makes no assumption about the cause: it records, for every step,
ego speed, the commanded action, which limit was active (IDM target / curve cap / visibility cap), the SafeTeacher time-to-collision, and every other vehicle within 25 m in the ego frame
(right, forward, speed, heading relative to ego). At termination it reports which vehicle was closest and whether it was ahead of, behind, or beside the ego.

    python -m bev_rl.adaptive.diagnose_visibility_failures --cases 12:3106 5:3127 5:3143 inf:3146
"""
import argparse
import math
import os

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="+", required=True, help="R:seed pairs, R in metres or inf"); ap.add_argument("--tail", type=int, default=14)
    a = ap.parse_args()
    from bev_rl.adaptive.caution import CautionEnv
    from bev_rl.adaptive.teacher_eval import build_teacher, curvature_cap_mps
    from bev_rl.adaptive.visibility_teacher import v_visibility
    env = CautionEnv(range(3100, 3150), [("clear", 0.0)], dropout=False, traffic_density=0.05, horizon=1600, sample_seed=7)
    for case in a.cases:
        Rs, ss = case.split(":"); R, seed = float(Rs), int(ss); v_vis = v_visibility(R, 0.5, 4.0, 2.0, 3.0)
        env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); idm, safe, target, _ = build_teacher("safe", env, 0.0); v_cmd = idm.NORMAL_SPEED; log = []
        while True:
            ag = env.md.agent; th = float(ag.heading_theta); right, fwd = np.array([math.sin(th), -math.cos(th)]), np.array([math.cos(th), math.sin(th)]); ego = np.asarray(ag.position, float)
            cap = curvature_cap_mps(env, float(ag.speed), 4.0); lim = min(v_cmd, cap * 3.6, v_vis * 3.6); idm.NORMAL_SPEED = idm.target_speed = lim
            act = np.clip(np.asarray(safe.act(env), np.float32)[:2], -1, 1); ttc = safe.last_ttc
            near = []
            for o in env.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
                if o is ag: continue
                d = np.asarray(o.position, float) - ego; x, y = float(d @ right), float(d @ fwd)
                if math.hypot(x, y) < 25.0: near.append((round(x, 1), round(y, 1), round(float(o.speed), 1), round(math.degrees((float(o.heading_theta) - th + math.pi) % (2 * math.pi) - math.pi))))
            which = "visibility" if v_vis * 3.6 < min(v_cmd, cap * 3.6) - 0.05 else ("curve" if cap * 3.6 < v_cmd - 0.05 else "idm")
            log.append(dict(v=round(float(ag.speed), 2), steer=round(float(act[0]), 2), thr=round(float(act[1]), 2), ttc=None if ttc is None else round(ttc, 2), limit=which, lim_ms=round(lim / 3.6, 2), curve_cap=round(min(cap, 99.0), 1), near=near))
            o, r, te, tr, info = env.step(act)
            if te or tr: break
        print(f"\n=== case R={Rs} m seed {seed}: outcome {info['outcome']} after {len(log)} steps ({0.1 * len(log):.0f} s), route completion {info.get('route_completion', float('nan')):.2f}")
        for k, e in enumerate(log[-a.tail:], start=len(log) - a.tail): print(f"  step {k:4d} v={e['v']:5.2f} steer={e['steer']:+.2f} thr={e['thr']:+.2f} ttc={e['ttc']} limit={e['limit']} (limit {e['lim_ms']} m/s, curve cap {e['curve_cap']}) near(right,fwd,speed,dHeading)={e['near']}")
        last = log[-1]["near"]; best = min(last, key=lambda n: math.hypot(n[0], n[1])) if last else None
        print("  closest vehicle at the end:", best, "-> ", ("BEHIND the ego" if best and best[1] < -0.5 else "AHEAD of the ego" if best and best[1] > 0.5 else "BESIDE the ego") if best else "none within 25 m")
        steps_ttc = sum(1 for e in log if e["ttc"] is not None and e["ttc"] < 2.4); print(f"  steps with TTC<2.4 s: {steps_ttc}; steps limited by visibility/curve/idm: {sum(e['limit']=='visibility' for e in log)}/{sum(e['limit']=='curve' for e in log)}/{sum(e['limit']=='idm' for e in log)}")
    env.close()


if __name__ == "__main__":
    main()
