"""LIVE caution sensor: estimates the caution level c in [0,1] from the camera picture alone, every step. No weather label, no simulator state.

    c_hat = clip( w . features(front camera frame) + b , 0, 1 )   then exponential smoothing over steps, then a small dead-zone (so clear weather gives exactly 0)

Features (cheap, on a 1/4-scale grayscale copy of the already weather-filtered front-left camera frame, the same frame the stereo/YOLO chain receives):
    log Laplacian variance (sharpness), mean gradient magnitude, RMS contrast, mean dark-channel (haze/fog veil), mean brightness, mean saturation, share of strong edges.
The weights are fitted OFFLINE (ridge regression) on frames from TRAINING seeds only, with target c = c_auto(weather label) = 1 - keep_prob from the measured YOLO retention table
(the same quantity the original caution model was given). The label is therefore used only as a regression target in the offline fit; at run time the estimator sees pixels only.

    python -m bev_rl.adaptive.visibility_sensor fit --seeds 2000-2039 --out bev_rl/adaptive/results/visibility_sensor
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
FEATURES = ["log_lapvar", "grad", "contrast", "dark", "bright", "sat", "edge"]
WEIGHTS_PATH = Path("bev_rl/adaptive/results/visibility_sensor/weights.json")


def features(frame):
    import cv2
    img = np.asarray(frame)
    if img.ndim == 3 and img.shape[2] == 4: img = img[:, :, :3]
    small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    g = small.mean(axis=2) if small.ndim == 3 else small
    lap = cv2.Laplacian(g, cv2.CV_32F); gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1); mag = np.sqrt(gx * gx + gy * gy)
    dark = cv2.erode(small.min(axis=2), np.ones((7, 7), np.uint8)) if small.ndim == 3 else g
    mx, mn = (small.max(axis=2), small.min(axis=2)) if small.ndim == 3 else (g, g)
    sat = float(((mx - mn) / (mx + 1e-3)).mean())
    return np.array([np.log(float(lap.var()) + 1e-8), float(mag.mean()), float(g.std()), float(dark.mean()), float(g.mean()), sat, float((mag > 0.15).mean())], np.float64)


class VisibilitySensor:
    """Run-time estimator. update(frame) -> smoothed c_hat. reset() at episode start."""

    def __init__(self, path=WEIGHTS_PATH, alpha=0.15):
        d = json.loads(Path(path).read_text()); self.mu, self.sd = np.array(d["mu"]), np.array(d["sd"]); self.w, self.b = np.array(d["w"]), float(d["b"])
        self.dead = float(d.get("dead_zone", 0.0)); self.alpha = alpha; self.c = None

    def reset(self): self.c = None

    def raw(self, frame): return float(np.clip(((features(frame) - self.mu) / self.sd) @ self.w + self.b, 0.0, 1.0))

    def update(self, frame):
        r = self.raw(frame); self.c = r if self.c is None else (1 - self.alpha) * self.c + self.alpha * r
        return 0.0 if self.c < self.dead else float(self.c)


def collect(a):
    import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
    from bev_policy.runtime import make_env
    from manual_drive_stereo_yolo_weather import frame_from_sensor
    from weather.weather_utils import apply_weather
    from bev_rl.adaptive.caution import c_auto, RET_PATH
    table = json.loads(RET_PATH.read_text()); lo, hi = map(int, a.seeds.split("-")); a.out.mkdir(parents=True, exist_ok=True); f = a.out / f"frames_{lo}-{hi}.jsonl"
    done = set()
    if f.exists(): done = {(r["seed"]) for r in map(json.loads, f.read_text().splitlines())}
    md = make_env(use_idm=False, render=False, traffic_density=0.1, horizon=300)
    W = [("clear", 0.0), ("fog", 0.3), ("fog", 0.6), ("fog", 1.0), ("rain", 0.2), ("rain", 0.45), ("rain", 0.6), ("rain", 1.0)]
    for seed in range(lo, hi + 1):
        if seed in done: continue
        md.reset(seed=seed); rows = []; zero = np.zeros(2, np.float32)
        for t in range(40):
            md.step(np.array([0.0, 0.3], np.float32))
            if t % 8 != 7: continue
            raw = frame_from_sensor(md, "front_left_camera")
            for kind, level in W:
                np.random.seed(20261114 + seed * 100003 + t)
                fr = raw if kind == "clear" else apply_weather(raw, kind, level)
                rows.append({"seed": seed, "t": t, "weather": f"{kind}:{level}", "c": c_auto(kind, level, table), "f": features(fr).tolist()})
        with open(f, "a") as fh: fh.write("\n".join(json.dumps({**r}) for r in rows) + "\n")
        print("seed", seed, len(rows), flush=True)
    md.close()


def fit(a):
    rows = [json.loads(l) for fn in sorted(a.out.glob("frames_*.jsonl")) for l in fn.read_text().splitlines() if l.strip()]
    X = np.array([r["f"] for r in rows]); y = np.array([r["c"] for r in rows]); seeds = np.array([r["seed"] for r in rows]); wl = np.array([r["weather"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    def ridge(Z, y, lam=1.0): A = Z.T @ Z + lam * np.eye(Z.shape[1]); w = np.linalg.solve(A, Z.T @ (y - y.mean())); return w, float(y.mean())
    us = np.unique(seeds); rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(us), 5); pred = np.zeros_like(y)
    for fo in folds:
        te = np.isin(seeds, fo); w, b = ridge(Z[~te], y[~te]); pred[te] = np.clip(Z[te] @ w + b, 0, 1)
    print("5-fold (by scene) estimate of c per weather:  true c | mean est | std est | p5 | p95")
    for k in dict.fromkeys(wl):
        m = wl == k; print(f"  {k:10s} {y[m][0]:.3f} | {pred[m].mean():.3f} | {pred[m].std():.3f} | {np.percentile(pred[m], 5):.3f} | {np.percentile(pred[m], 95):.3f}")
    print("overall MAE", float(np.abs(pred - y).mean()), "corr", float(np.corrcoef(pred, y)[0, 1]))
    clear = pred[wl == "clear:0.0"]; dead = float(np.percentile(clear, 99)); print("dead zone (99th pct of clear estimates):", dead)
    w, b = ridge(Z, y); (a.out / "weights.json").write_text(json.dumps({"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": b, "dead_zone": dead, "n_frames": int(len(y)), "fit_seeds": f"{int(us.min())}-{int(us.max())}"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["collect", "fit"]); ap.add_argument("--seeds", default="2000-2039"); ap.add_argument("--out", type=Path, default=Path("bev_rl/adaptive/results/visibility_sensor"))
    a = ap.parse_args(); {"collect": collect, "fit": fit}[a.cmd](a)
