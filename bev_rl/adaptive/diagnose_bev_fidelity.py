"""Does the BEV the policy receives represent the TRUE position of the vehicle relative to the ego? Replay a model and, for the last steps before the episode ends, compare
  true gap   : distance between the ORIENTED rectangles of the ego and the nearest vehicle (MetaDrive pose, heading, size)           [metres]
  BEV gap    : distance between the ego block and the nearest vehicle block AS DRAWN IN THE BEV GRID the model was given (0.3125 m cells)   [metres]
plus the vehicle's heading relative to the ego, its lateral/forward offset, and whether the vehicle is inside the BEV window. Also reports what the ego collided with.
    python -m bev_rl.adaptive.diagnose_bev_fidelity --arm base_a --seeds 1012 1039 1040 --last 14
"""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np


def rect(center, heading, length, width):
    c, s = np.cos(heading), np.sin(heading); f, r = np.array([c, s]), np.array([-s, c])
    return np.array([center + f * length / 2 + r * width / 2, center + f * length / 2 - r * width / 2, center - f * length / 2 - r * width / 2, center - f * length / 2 + r * width / 2])


def poly_dist(A, B):
    """min distance between two convex quadrilaterals (0 if they overlap); exact via shapely when available."""
    try:
        from shapely.geometry import Polygon
        return float(Polygon(A).distance(Polygon(B)))
    except Exception:
        def seg(p, q, r):
            d = q - p; t = np.clip(((r - p) @ d) / max(d @ d, 1e-12), 0, 1); return np.linalg.norm(p + t * d - r)
        best = 1e9
        for P, Q in ((A, B), (B, A)):
            for i in range(4):
                for r in Q: best = min(best, seg(P[i], P[(i + 1) % 4], r))
        return best


def bev_gap_m(bev, mpc=0.3125):
    ego = np.argwhere(np.abs(bev - 0.9) < 0.05); veh = np.argwhere(bev >= 0.95)
    if len(ego) == 0 or len(veh) == 0: return None
    r0, r1, c0, c1 = ego[:, 0].min(), ego[:, 0].max(), ego[:, 1].min(), ego[:, 1].max()
    dr = np.maximum(np.maximum(r0 - veh[:, 0], veh[:, 0] - r1), 0); dc = np.maximum(np.maximum(c0 - veh[:, 1], veh[:, 1] - c1), 0)
    return float(np.sqrt((dr * mpc) ** 2 + (dc * mpc) ** 2).min())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="base_a"); ap.add_argument("--seeds", nargs="+", type=int, default=[1012, 1039, 1040]); ap.add_argument("--last", type=int, default=14)
    a = ap.parse_args()
    import torch
    from bev_rl.adaptive.degraded_env import DegradedEnv
    from bev_rl.adaptive.env import SCALE
    from bev_rl.adaptive.record_videos import build
    env = DegradedEnv(a.seeds, [("clear", 0.0)], mode="dropout", traffic_density=0.05, horizon=800, sample_seed=7); P = build(a.arm, torch, env)
    for seed in a.seeds:
        o, _ = env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)}); env.drng = np.random.default_rng(seed * 7 + 1); hist = []; t = 0
        while True:
            raw = (o[:6] * SCALE).astype(np.float32)
            bev = env.asm.build_bev_grid(env.truth_objects()); sc = ((raw - P["mu"]) / P["sd"]).astype(np.float32)
            with torch.no_grad(): act = P["net"](torch.from_numpy(bev[None, None].astype(np.float32)), torch.from_numpy(sc[None])).squeeze(0).numpy()
            ag = env.md.agent; pe = np.asarray(ag.position, float); the = float(ag.heading_theta)
            E = rect(pe, the, float(getattr(ag, "LENGTH", 4.5)), float(getattr(ag, "WIDTH", 1.85))); rows = []
            for ob in env.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
                if ob is ag: continue
                po = np.asarray(ob.position, float); d = po - pe
                if np.hypot(*d) > 40: continue
                right, fwd = np.array([np.sin(the), -np.cos(the)]), np.array([np.cos(the), np.sin(the)])
                O = rect(po, float(ob.heading_theta), float(getattr(ob, "LENGTH", 4.5)), float(getattr(ob, "WIDTH", 1.85)))
                rel = np.degrees(((float(ob.heading_theta) - the + np.pi) % (2 * np.pi)) - np.pi)
                rows.append((poly_dist(E, O), float(d @ right), float(d @ fwd), rel, float(np.linalg.norm(np.asarray(ob.velocity, float) - np.asarray(ag.velocity, float))), type(ob).__name__))
            rows.sort(key=lambda r: r[0]); hist.append((t, rows[0] if rows else None, bev_gap_m(bev), float(ag.speed)))
            o, r, te, tr, info = env.step(act); t += 1
            if te or tr: break
        flags = [k for k in ("crash_vehicle", "crash_object", "crash_building", "crash_human", "out_of_road", "arrive_dest") if info.get(k)]
        print(f"\n=== {a.arm}  seed {seed}  outcome {info['outcome']}  steps {t}  flags {flags}")
        print(" step | nearest vehicle: true gap | x_right  y_fwd | heading rel ego | closing m/s || BEV-drawn gap | ego speed")
        for st, row, bg, sp in hist[-a.last:]:
            if row is None: print(f" {st:4d} | none within 40 m"); continue
            print(f" {st:4d} | {row[0]:7.2f} m           | {row[1]:6.1f}  {row[2]:6.1f} | {row[3]:8.0f} deg     | {row[4]:6.1f}      || {('%.2f m' % bg) if bg is not None else 'not in BEV':>12s} | {sp:4.1f}")
        env.md.engine  # keep reference
    env.close()


if __name__ == "__main__":
    main()
