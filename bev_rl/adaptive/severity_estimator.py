"""IMAGE-BASED WEATHER-SEVERITY ESTIMATOR (kind classifier + separate fog / rain intensity regressors) -> caution c.   NEW FILE (2026-11-16).  Replaces the retired retention-based sensor.

Input   the SEVEN existing image features (visibility_sensor.features) of the final weather-processed front-left frame, nothing else.
Model   kind: softmax (clear / fog / rain), balanced class weights, L2 lambda = 1e-2.   level: one ridge regressor per kind (lambda = 1) fitted on clear (level 0) + that kind's frames, clipped to [0, 1].
        All hyper-parameters are fixed a priori (as in the feasibility check); nothing is tuned on held-out data.  The probes (fog 0.45, rain 0.3, fog 0.6, rain 0.45) never enter a fit.
Caution identity mapping (documented design assumption; fog and rain levels are generator-normalised, NOT physically equal):  clear -> 0,  fog -> predicted fog level,  rain -> predicted rain level.
Live filter (SeverityEstimator.update): exponential moving average of the kind probabilities (alpha_p) -> hard kind = argmax; per-frame level of that kind (0 for clear) -> EMA (alpha_c).
        Interface identical to VisibilitySensor (reset(), update(frame) -> c_hat) so camera_dagger.BuilderEnv uses it unchanged.

    python -m bev_rl.adaptive.severity_estimator fit  --out bev_rl/adaptive/results/severity_estimator/estimator.json
    python -m bev_rl.adaptive.severity_estimator eval --model .../estimator.json            (held-out streams, seeds 9040-9049; report, never fits)
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

KINDS = ["clear", "fog", "rain"]
PROBES = {("fog", 0.45), ("rain", 0.3), ("fog", 0.6), ("rain", 0.45)}
LAM_CLS, LAM_REG = 1e-2, 1.0
RES = Path("bev_rl/adaptive/results")
ALPHA_P, ALPHA_C = 0.15, 0.15


# ------------------------------------------------------------------------------------------------------------------------------------------- fitting
def fit_softmax(Z, y, w, lam, K=3):
    from scipy.optimize import minimize
    n, d = Z.shape; Zb = np.c_[Z, np.ones(n)]; Y = np.eye(K)[y]
    def f(v):
        W = v.reshape(d + 1, K); S = Zb @ W; S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True)
        L = -(w * np.log(P[np.arange(n), y] + 1e-12)).sum() / w.sum() + 0.5 * lam * (W[:-1] ** 2).sum()
        G = Zb.T @ (w[:, None] * (P - Y)) / w.sum(); G[:-1] += lam * W[:-1]; return L, G.ravel()
    return minimize(f, np.zeros((d + 1) * K), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x.reshape(d + 1, K)


def fit_ridge(Z, y, lam):
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); w = np.linalg.solve(A, Z.T @ (y - y.mean())); return w, float(y.mean())


def fit_model(X, kind, level):
    """X (n,7) raw features, kind (n,) in 0/1/2, level (n,). Probe rows must already be removed."""
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    w = np.zeros(len(kind))
    for k in range(3): w[kind == k] = 1.0 / max(1, (kind == k).sum())
    W = fit_softmax(Z, kind, w, LAM_CLS); reg = {}
    for k in (1, 2):
        m = (kind == 0) | (kind == k); reg[k] = fit_ridge(Z[m], level[m], LAM_REG)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "W": W.tolist(), "reg_w": {str(k): reg[k][0].tolist() for k in reg}, "reg_b": {str(k): reg[k][1] for k in reg}, "n_train": int(len(kind)),
            "lam_cls": LAM_CLS, "lam_reg": LAM_REG}


def predict_frames(M, X):
    """-> kind probabilities (n,3), per-kind level estimates (n,3) (column 0 = 0)."""
    Z = (np.asarray(X, float) - np.asarray(M["mu"])) / np.asarray(M["sd"]); S = np.c_[Z, np.ones(len(Z))] @ np.asarray(M["W"]); S -= S.max(1, keepdims=True); P = np.exp(S); P /= P.sum(1, keepdims=True)
    L = np.zeros((len(Z), 3))
    for k in (1, 2): L[:, k] = np.clip(Z @ np.asarray(M["reg_w"][str(k)]) + M["reg_b"][str(k)], 0, 1)
    return P, L


class SeverityEstimator:
    """Live caution estimator (drop-in for VisibilitySensor): reset(), update(front_left_frame) -> c_hat. Also exposes .kind_hat, .p, .c_frame for logging."""
    def __init__(self, path=RES / "severity_estimator" / "estimator.json", alpha_p=ALPHA_P, alpha_c=ALPHA_C):
        self.M = json.loads(Path(path).read_text()); self.ap, self.ac = alpha_p, alpha_c; self.reset()

    def reset(self): self.p = None; self.c = None; self.kind_hat = 0; self.c_frame = 0.0

    def update(self, frame):
        from bev_rl.adaptive.visibility_sensor import features
        P, L = predict_frames(self.M, features(frame)[None])
        self.p = P[0] if self.p is None else (1 - self.ap) * self.p + self.ap * P[0]; self.kind_hat = int(self.p.argmax())
        self.c_frame = float(L[0, self.kind_hat]); self.c = self.c_frame if self.c is None else (1 - self.ac) * self.c + self.ac * self.c_frame
        return float(self.c)


# ------------------------------------------------------------------------------------------------------------------------------------------- data
def parse(c): k, l = c.split(":"); return k, float(l)


def load_dagger(root=RES / "camera_dagger_severity" / "e2e_sevB"):
    """Whole-route frames of the Model B DAgger episodes (every recorded step; weather = the episode's weather). group = round_seed so episodes never split."""
    X, kind, lvl, grp, t = [], [], [], [], []
    for f in sorted(Path(root).glob("round*/*.npz")):
        z = np.load(f); F = z["F"]; k = str(z["kind"]); l = float(z["level"])
        if (k, l) in PROBES: continue
        X.append(F); kind += [KINDS.index(k)] * len(F); lvl += [l] * len(F); grp += [f"{f.parent.name}_{int(z['seed'])}"] * len(F); t += list(range(len(F)))
    return (np.concatenate(X).astype(float), np.array(kind), np.array(lvl), np.array(grp), np.array(t)) if X else None


def load_preliminary():
    """The existing 28 retention-pass episodes (seeds 9000-9027): retention_features only, non-probe conditions. Preliminary material (limited scene diversity)."""
    X, kind, lvl, grp, t = [], [], [], [], []
    for fn in sorted(glob.glob(str(RES / "retention_camera" / "episodes" / "s90[0-2]*.json"))):
        e = json.load(open(fn))
        if e.get("status") != "complete" or not (9000 <= e["seed"] <= 9029): continue
        for r in e["retention_features"]:
            for c, f in r["f"].items():
                k, l = parse(c)
                if (k, l) in PROBES: continue
                X.append(f); kind.append(KINDS.index(k)); lvl.append(l); grp.append(f"prelim_{e['seed']}"); t.append(r["t"])
    return np.array(X, float), np.array(kind), np.array(lvl), np.array(grp), np.array(t)


def load_heldout():
    """Held-out whole-route streams 9040-9049 -> {seed: (t (T,), cond_names (11,), feat (T,11,7), valid (T,11))}. Clear at every step; weather cells only at even steps (unsampled cells are invalid)."""
    out = {}
    for f in sorted(glob.glob(str(RES / "retention_camera" / "streams" / "s904*_stream.npz"))):
        z = np.load(f, allow_pickle=True); out[int(Path(f).stem[1:5])] = (z["t"], [str(c) for c in z["cond_names"]], z["feat"], z["valid"])
    return out


def _cluster_ci(groups, values_fn, B=2000, seed=20261116):
    rng = np.random.default_rng(seed); u = np.unique(groups); vals = []
    for _ in range(B):
        pick = rng.choice(u, len(u)); vals.append(values_fn(np.concatenate([np.where(groups == g)[0] for g in pick])))
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def eval_cmd(a):
    """Held-out evaluation (never fits). Per-frame kind/level accuracy per condition + simulation of the live EMA filter on the streams (weather stride 2 -> filter alpha adapted)."""
    M = json.loads(a.model.read_text()); H = load_heldout(); rows = []
    for seed, (t, names, feat, valid) in H.items():
        for k, cn in enumerate(names):
            kd, lv = parse(cn); m = valid[:, k]
            if not m.any(): continue
            P, L = predict_frames(M, feat[m, k]); kh = P.argmax(1)
            for i in range(len(P)): rows.append((seed, int(t[m][i]), cn, KINDS.index(kd), lv, kh[i], L[i, kh[i]] if kh[i] else 0.0))
    R = rows; seed = np.array([r[0] for r in R]); cond = np.array([r[2] for r in R]); kd = np.array([r[3] for r in R]); lv = np.array([r[4] for r in R]); kh = np.array([r[5] for r in R]); ch = np.array([r[6] for r in R])
    out = {"model": str(a.model), "n_frames": len(R), "per_condition": {}}
    for cn in sorted(set(cond), key=lambda c: (parse(c)[0], parse(c)[1])):
        m = cond == cn; k0 = kd[m][0]; probe = parse(cn) in PROBES
        e = {"n": int(m.sum()), "probe": probe, "kind_recall": float((kh[m] == k0).mean()), "c_hat_mean": float(ch[m].mean()), "c_true": float(lv[m][0]), "bias": float(ch[m].mean() - lv[m][0]),
             "mae": float(np.abs(ch[m] - lv[m]).mean())}
        out["per_condition"][cn] = e
    for kind_i, kn in ((1, "fog"), (2, "rain")):                      # separate intensity MAE over the kind's own conditions, true kind known, kind-correct frames and all frames
        m = kd == kind_i; out[f"{kn}_mae_all"] = float(np.abs(ch[m] - lv[m]).mean()); mm = m & (kh == kind_i); out[f"{kn}_mae_kind_correct"] = float(np.abs(ch[mm] - lv[mm]).mean())
    cl = kd == 0; out["clear_false_caution_frame_rate"] = float((ch[cl] > 0.05).mean()); out["clear_c_hat_mean"] = float(ch[cl].mean())
    non_probe = np.array([parse(c) not in PROBES for c in cond]); out["kind_accuracy_nonprobe"] = float((kh == kd)[non_probe].mean()); out["kind_accuracy_probe"] = float((kh == kd)[~non_probe].mean())
    out["macro_recall_nonprobe"] = float(np.mean([(kh[non_probe & (kd == k)] == k).mean() for k in range(3)]))
    # live filter simulation per (seed, condition): alpha adapted to the 5 Hz update (stride-2) so the time constant equals the live 10 Hz filter
    ap_, ac_ = 1 - (1 - ALPHA_P) ** 2, 1 - (1 - ALPHA_C) ** 2; sim = {}
    for seed_, (t, names, feat, valid) in H.items():
        for k, cn in enumerate(names):
            kd_, lv_ = parse(cn); idx = np.where(valid[:, k])[0]
            if len(idx) == 0: continue
            P, L = predict_frames(M, feat[idx, k]); p = None; c = None; cs = []
            for i in range(len(P)):
                p = P[i] if p is None else (1 - ap_) * p + ap_ * P[i]; kh_ = int(p.argmax()); cf = float(L[i, kh_]) if kh_ else 0.0; c = cf if c is None else (1 - ac_) * c + ac_ * cf; cs.append(c)
            cs = np.array(cs); sim.setdefault(cn, []).append((np.abs(cs - lv_).mean(), cs[len(cs) // 4:].mean() - lv_, float(cs.std())))
    out["filtered"] = {cn: {"mae": float(np.mean([x[0] for x in v])), "bias_after_first_quarter": float(np.mean([x[1] for x in v])), "within_episode_sd": float(np.mean([x[2] for x in v]))} for cn, v in sim.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_condition", "filtered")}, indent=1))
    for cn, e in out["per_condition"].items(): print(f"{cn:10s} n={e['n']:5d} probe={int(e['probe'])} kind_recall={e['kind_recall']:.3f} c_hat={e['c_hat_mean']:.3f} true={e['c_true']:.2f} bias={e['bias']:+.3f} MAE={e['mae']:.3f}  filtered MAE={out['filtered'][cn]['mae']:.3f}")


def fit_cmd(a):
    parts = []
    d = load_dagger()
    if d is not None: parts.append(d)
    if not a.no_prelim: parts.append(load_preliminary())
    X = np.concatenate([p[0] for p in parts]); kind = np.concatenate([p[1] for p in parts]); lvl = np.concatenate([p[2] for p in parts])
    M = fit_model(X, kind, lvl); M["sources"] = {"dagger_episodes": int(len(set(d[3]))) if d is not None else 0, "prelim_frames": 0 if a.no_prelim else int(len(parts[-1][0]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists(): raise SystemExit(f"{a.out} exists; refusing to overwrite (choose a new --out)")
    a.out.write_text(json.dumps(M)); print("fitted on", len(X), "frames ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fit", "eval"]); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--no-prelim", action="store_true"); ap.add_argument("--model", type=Path)
    a = ap.parse_args(); {"fit": fit_cmd, "eval": eval_cmd}[a.cmd](a)
