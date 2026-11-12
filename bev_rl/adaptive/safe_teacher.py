"""Collision-aware teacher: MetaDrive IDM (lane keeping + same-lane car following) + a privileged predictive safety layer.

Why: measured 2026-11-04 (expert_ceiling_lane_change_ablation, diagnose_teacher_crash_causes): stock IDM crashes 45% of the time at traffic
density 0.2, almost always into vehicles that MERGE / CUT IN / CROSS from the side, which IDM (same-lane only) never reacts to. A student
cloned from it inherits the ceiling. This teacher predicts every other vehicle with a constant-velocity model over `horizon_s` and brakes when
the ego's footprint (moving along its current velocity) would overlap another vehicle's footprint within `t_brake_s`.
Uses simulator ground truth (position, velocity, size) => legitimate for a TEACHER only; the student still sees the BEV.
"""
import numpy as np


class SafeTeacher:
    def __init__(self, idm, horizon_s=3.0, t_hard_s=1.3, t_soft_s=2.4, lat_margin=0.35, long_margin=1.0, dt=0.1):
        self.idm, self.horizon, self.t_hard, self.t_soft = idm, horizon_s, t_hard_s, t_soft_s
        self.lat_m, self.long_m, self.dt = lat_margin, long_margin, dt
        self.last_ttc = None

    @staticmethod
    def _extent(o, ux, uy):
        """half-extent of vehicle o's rectangle projected on the unit axis (ux,uy)."""
        L, W = float(getattr(o, "LENGTH", 4.5)), float(getattr(o, "WIDTH", 1.85)); th = float(o.heading_theta)
        fx, fy, rx, ry = np.cos(th), np.sin(th), -np.sin(th), np.cos(th)
        return 0.5 * L * abs(fx * ux + fy * uy) + 0.5 * W * abs(rx * ux + ry * uy)

    def ttc(self, env):
        ag = env.md.agent; pe = np.asarray(ag.position, float); ve = np.asarray(ag.velocity, float)
        th = float(ag.heading_theta); fw = np.array([np.cos(th), np.sin(th)]); rt = np.array([-np.sin(th), np.cos(th)])
        hl_e, hw_e = 0.5 * float(getattr(ag, "LENGTH", 4.5)), 0.5 * float(getattr(ag, "WIDTH", 1.85))
        best = None
        for o in env.md.engine.get_objects(lambda o: hasattr(o, "speed")).values():
            if o is ag: continue
            d0 = np.asarray(o.position, float) - pe
            if np.hypot(*d0) > 45.0: continue
            vr = np.asarray(o.velocity, float) - ve                      # relative velocity (ego frame fixed at current heading)
            hl_o, hw_o = self._extent(o, *fw), self._extent(o, *rt)
            for t in np.arange(0.0, self.horizon + 1e-6, self.dt):
                d = d0 + vr * t
                if abs(d @ fw) < hl_e + hl_o + self.long_m and abs(d @ rt) < hw_e + hw_o + self.lat_m:
                    if best is None or t < best: best = t
                    break
        return best

    def act(self, env):
        a = np.clip(np.asarray(self.idm.act(), np.float32)[:2], -1, 1)
        t = self.ttc(env); self.last_ttc = t
        if t is not None:
            if t < self.t_hard: a[1] = -1.0
            elif t < self.t_soft: a[1] = min(a[1], -0.2 - 0.8 * (self.t_soft - t) / (self.t_soft - self.t_hard))
        return a
