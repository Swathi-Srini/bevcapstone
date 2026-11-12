"""Policy-side environment with CALIBRATED perception degradation (replaces the dropout-only proxy of caution.py).

Why: weather acts on the camera pipeline in two ways, both measured (measure_depth_error.py -> fit_depth_error_model.py -> results/depth_error_model.json):
  (a) a vehicle is not detected, or its box has no valid stereo depth ('stereo_sgm_invalid')            -> the vehicle disappears from the BEV
  (b) a detected vehicle gets a WRONG RANGE (stereo mismatches)                                       -> the vehicle appears at the wrong place in the BEV
Student BEV = overlay( build_bev_grid( degraded(truth_objects) ) ). The teacher keeps ground truth. Nothing in bev_state/ or the YOLO module is modified.

Per object in the BEV window (ego frame x right, y forward):
  * inside the front camera field of view (|x| <= (y + 2.0) * tan(30 deg), y > 0.5): sample (detected & valid depth, relative range error) from the measured cell for the current
    (kind, level) and range bin of the camera-forward range r = y - L/2 - 2.0; the estimated range is r*(1+rel); lateral position scales with range (x' = x*(r+d)/r), forward y' = y + d.
    Levels between measured conditions use stochastic interpolation (sample the lower/upper measured cell with probability by distance).
  * outside the front FOV (side/rear cameras use MONOCULAR ground-plane depth, which was NOT measured): only the measured detection retention is applied (caution.keep_prob) - declared limitation.
Errors are i.i.d. per step per object, matching the augmentation (fresh random draw every frame).
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from bev_rl.adaptive.caution import CautionEnv, KEEP, keep_prob
from bev_rl.adaptive.env import SCALE, weather_vector
from bev_rl.adaptive.spec_bev import overlay_github_bev

MODEL_PATH = Path("bev_rl/adaptive/results/depth_error_model.json")
CAM_FWD = 2.0
TAN_HALF_FOV = float(np.tan(np.radians(30.0)))


class DepthModel:
    def __init__(self, path=MODEL_PATH):
        d = json.loads(Path(path).read_text()); self.bins = np.array(d["range_bins_m"]); self.cells = {}
        for k, c in d["cells"].items():
            n = np.array(c["n"], float); pd = np.array([np.nan if v is None else v for v in c["p_detect"]], float); pv = np.array([np.nan if v is None else v for v in c["p_valid"]], float)
            def fill(a):
                good = ~np.isnan(a)
                return np.where(good, a, (a[good] * n[good]).sum() / max(1.0, n[good].sum()) if good.any() else 1.0)
            self.cells[k] = {"pd": fill(pd), "pv": fill(pv), "rel": [np.array(r if r else c["pooled_rel_err"], float) for r in c["rel_err"]]}
        self.levels = {}
        for k in self.cells:
            kind, lvl = k.split(":")
            if kind != "none": self.levels.setdefault(kind, []).append(float(lvl))
        for v in self.levels.values(): v.sort()

    def _key(self, kind, level, rng):
        if kind == "clear" or level <= 0 or kind not in self.levels: return "none:0.0"
        lv = [0.0] + self.levels[kind]
        if level >= lv[-1]: return f"{kind}:{lv[-1]}"
        i = int(np.searchsorted(lv, level, side="right")) - 1; lo, hi = lv[i], lv[i + 1]
        pick = hi if rng.random() < (level - lo) / (hi - lo) else lo
        return "none:0.0" if pick == 0.0 else f"{kind}:{pick}"

    def sample(self, kind, level, rcam, rng):
        """-> (visible_in_bev: bool, relative range error)."""
        c = self.cells[self._key(kind, level, rng)]; b = int(min(len(self.bins) - 2, max(0, np.searchsorted(self.bins, rcam, side="right") - 1)))
        if rng.random() >= c["pd"][b] or rng.random() >= c["pv"][b]: return False, 0.0
        r = c["rel"][b]
        return True, float(r[int(rng.integers(len(r)))])


class DegradedEnv(CautionEnv):
    """mode: 'calibrated' (measured depth/detection model), 'dropout' (old keep-prob only), 'none' (perfect perception)."""

    def __init__(self, seeds, conditions, mode="calibrated", traffic_density=0.05, **kw):
        super().__init__(seeds, conditions, dropout=(mode == "dropout"), traffic_density=traffic_density, **kw)
        self.mode = mode
        self.depth = DepthModel() if mode == "calibrated" else None

    def degrade(self, objs):
        if self.mode == "none": return objs
        kind, level = self.weather
        out = []
        for o in objs:
            x, y = o.x_right_m, o.y_forward_m
            if self.mode == "calibrated" and y > 0.5 and abs(x) <= (y + CAM_FWD) * TAN_HALF_FOV:
                r = max(0.5, y - 0.5 * o.length_m - CAM_FWD)
                ok, rel = self.depth.sample(kind, level, r, self.drng)
                if not ok: continue
                d = rel * r
                if r + d < 1.0: continue                           # the pipeline discards depth < 1 m
                x, y = x * (r + d) / r, y + d
            elif self.drng.random() >= keep_prob(kind, level, self.table):
                continue
            if -10.0 <= x <= 10.0 and -2.5 <= y <= 17.5: out.append(replace(o, x_right_m=float(x), y_forward_m=float(y)))
        return out

    def student_bev(self):
        return overlay_github_bev(self.asm.build_bev_grid(self.degrade(self.truth_objects())), self.md)

    def min_gap_m(self):
        """Privileged safety metric: bumper-to-bumper gap to the nearest vehicle ahead in the ego corridor (30 m if none)."""
        g = 30.0
        for o in self.truth_objects():
            if o.y_forward_m > 0 and abs(o.x_right_m) < 2.0: g = min(g, o.y_forward_m - 0.5 * o.length_m - 2.3)
        return float(g)


def make_safe_expert(env):
    from metadrive.policy.idm_policy import IDMPolicy
    from bev_rl.adaptive.safe_teacher import SafeTeacher
    idm = IDMPolicy(env.md.agent, env.md.engine.global_seed); idm.enable_lane_change = False
    return SafeTeacher(idm, t_soft_s=2.4, t_hard_s=1.3)


def run_episode(env, net, mu, sd, torch, seed, weather, adaptive, beta=0.0, noise=0.0, rng=None, collect=False):
    """One episode. adaptive=True feeds weather_vector(kind, level) to the network's weather branch. collect=True returns (B,S,W,Y,info) with SafeTeacher labels (ground truth)."""
    o, _ = env.reset(options={"scenario_seed": seed, "weather": weather}); env.drng = np.random.default_rng(seed * 7 + 1)
    pol = make_safe_expert(env) if collect else None
    wv = weather_vector(*weather); B, S, W, Y = [], [], [], []; min_gap = 30.0
    while True:
        bev = env.student_bev(); raw = (o[:6] * SCALE).astype(np.float32); ea = None
        if collect:
            ea = np.clip(np.asarray(pol.act(env), np.float32)[:2], -1, 1)
            B.append(np.rint(bev * 10).astype(np.int8)); S.append(raw); W.append(wv); Y.append(ea)
        a_net = None
        if net is not None and not (collect and rng.random() < beta):
            b = torch.from_numpy(bev[None, None].astype(np.float32)); s = torch.from_numpy((((raw - mu) / sd) * KEEP)[None].astype(np.float32))
            with torch.no_grad():
                a_net = (net(b, s, torch.from_numpy(wv[None])) if adaptive else net(b, s)).squeeze(0).numpy()
        a = ea if a_net is None else a_net
        if noise > 0: a = np.clip(a + rng.normal(0, noise, 2), -1, 1).astype(np.float32)
        o, r, te, tr, info = env.step(a); min_gap = min(min_gap, env.min_gap_m())
        if te or tr:
            info["min_gap_m"] = min_gap
            return B, S, W, Y, info
