"""Reliability-Coupled Caution Conditioning (policy-side contribution).

Idea
    One BEV policy takes a caution scalar c in [0,1] as an extra input. The IDM teacher is made caution-aware (lower desired
    speed, larger desired gap as c grows), so a single network spans a speed / safety / energy trade-off. At run time c is NOT
    chosen by hand: it is derived from the MEASURED detector reliability under the current weather,
        c_auto(weather) = 1 - keep_prob(weather)         (keep_prob = measured YOLO detection retention, capped at 1)
    so the car becomes cautious exactly when its perception stops being trustworthy.

Training signal (privileged teacher, degraded student)
    The teacher IDM sees every vehicle (simulator truth); the student sees a BEV in which each vehicle is dropped with
    probability 1 - keep_prob (measured, see ppo_bev.keep_prob). The student cannot see what it missed, so the only way to be
    safe is to be cautious in advance; c tells it when.

No dynamics coupling: friction is held at the MetaDrive default in every weather (user rule: no weather physics).
Uses the unmodified GitHub grid builder (build_bev_grid) and the already-measured retention table. No camera/YOLO code is run.
"""
import json
from pathlib import Path

import numpy as np

from bev_rl.adaptive.env import AdaptiveEnv, SCALE, MU0

RET_PATH = Path("bev_rl/adaptive/results/retention_table.json")
KEEP = np.array([1, 1, 0, 0, 0, 1], np.float32)          # ov_split design: overlay BEV carries lateral/heading/curvature; scalar keeps speed, progress, goal


def keep_prob(kind, level, table):
    if kind == "clear" or level <= 0:
        return 1.0
    pts = sorted((float(k.split(":")[1]), min(1.0, v["retention"])) for k, v in table.items() if k.startswith(kind + ":"))
    xs, ys = [0.0] + [p[0] for p in pts], [1.0] + [p[1] for p in pts]
    return float(np.interp(level, xs, ys))


def c_auto(kind, level, table):
    """Caution implied by measured perception reliability."""
    return float(np.clip(1.0 - keep_prob(kind, level, table), 0.0, 1.0))


# expert caution mapping: c=0 -> stock IDM (30 km/h, 10 m gap); c=1 -> 20 km/h, 25 m gap
def expert_params(c):
    return 30.0 * (1.0 - c / 3.0), 10.0 + 15.0 * c


class CautionEnv(AdaptiveEnv):
    """Fast MetaDrive env. Student BEV = overlay + detector dropout; teacher sees all vehicles; friction constant."""

    def __init__(self, seeds, conditions, dropout=True, traffic_density=0.1, **kw):
        super().__init__(seeds, conditions, weather_input="zeroed", traffic_density=traffic_density, **kw)
        self.overlay = True
        self.dropout = dropout
        self.table = json.loads(RET_PATH.read_text())
        self.drng = np.random.default_rng(kw.get("sample_seed", 0) + 17)
        self._cache_free_space()

    def _cache_free_space(self):
        """Speed-up (instance-level; bev_state/ untouched): the camera free-space mask depends only on the static config, never on objects,
        so compute it once with the original routine and copy it afterwards. Verified bit-identical in tests (see caution_selftest)."""
        from bev_state import BEVStateAssembler
        cfg = self.asm.config
        tmpl = np.full((cfg.grid_size, cfg.grid_size), cfg.values.unknown, np.float32)
        BEVStateAssembler._mark_visible_free_space(self.asm, tmpl)
        self.asm._mark_visible_free_space = lambda grid: np.copyto(grid, tmpl)

    def reset(self, *, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        for w in self.md.agent.system.getWheels():           # undo AdaptiveEnv's rain->friction coupling: no weather physics
            w.setFrictionSlip(MU0)
        self.mu = MU0
        return o, i

    def student_bev(self):
        from bev_rl.adaptive.spec_bev import overlay_github_bev
        p = keep_prob(*self.weather, self.table) if self.dropout else 1.0
        objs = [o for o in self.truth_objects() if self.drng.random() < p]
        return overlay_github_bev(self.asm.build_bev_grid(objs), self.md)

    def make_expert(self, c):
        from metadrive.policy.idm_policy import IDMPolicy
        pol = IDMPolicy(self.md.agent, self.md.engine.global_seed)
        pol.NORMAL_SPEED, pol.DISTANCE_WANTED = expert_params(c)
        pol.target_speed = pol.NORMAL_SPEED
        return pol


def prep(torch, bev, raw, c, mu, sd):
    b = torch.from_numpy(bev[None, None].astype(np.float32))
    s = torch.from_numpy((((raw - mu) / sd) * KEEP)[None].astype(np.float32))
    return b, s, torch.tensor([[c]], dtype=torch.float32)


def episode(env, net, mu, sd, torch, seed, weather, c, beta=0.0, noise=0.0, rng=None, collect=False):
    """One episode. c = caution fed to the student (and used by the teacher for labels). beta = prob. of executing the teacher action."""
    o, _ = env.reset(options={"scenario_seed": seed, "weather": weather})
    pol = env.make_expert(c) if collect else None
    B, S, C, Y = [], [], [], []
    while True:
        bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32)
        a_net = None
        if collect:
            ea = np.clip(np.asarray(pol.act(), np.float32)[:2], -1, 1)
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); C.append(c); Y.append(ea)
        if net is not None and not (collect and rng.random() < beta):
            with torch.no_grad():
                a_net = net(*prep(torch, bev, raw, c, mu, sd)).squeeze(0).numpy()
        a = ea if a_net is None else a_net
        if noise > 0:
            a = np.clip(a + rng.normal(0, noise, 2), -1, 1).astype(np.float32)
        o, r, te, tr, info = env.step(a)
        if te or tr:
            return B, S, C, Y, info
