"""SENSOR VALIDATION (not tuning): apply the FROZEN weights (results/visibility_sensor/weights.json, fitted on seeds 2000-2041) to frames of unseen scenes (seeds 3100-3139).
Nothing here refits or adjusts anything. Per-weather error of the raw estimate and of the estimate after the frozen dead-zone."""
import glob, json, numpy as np
from pathlib import Path
W = json.loads(Path("bev_rl/adaptive/results/visibility_sensor/weights.json").read_text()); mu, sd, w, b, dead = map(np.array, (W["mu"], W["sd"], W["w"], W["b"], W["dead_zone"]))
rows = [json.loads(l) for f in sorted(glob.glob("bev_rl/adaptive/results/visibility_sensor_test/frames_*.jsonl")) for l in open(f) if l.strip()]
X = np.array([r["f"] for r in rows]); y = np.array([r["c"] for r in rows]); wl = np.array([r["weather"] for r in rows]); seeds = sorted({r["seed"] for r in rows})
pred = np.clip(((X - mu) / sd) @ w + b, 0, 1); after = np.where(pred < float(dead), 0.0, pred)
print(f"SENSOR VALIDATION on unseen scenes: seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} scenes, {len(rows)} frames); weights fitted on seeds {W['fit_seeds']}; nothing tuned on these frames")
print("weather | target c | mean est | std | bias | MAE | MAE after dead-zone | share of frames reading 0 after dead-zone")
for k in dict.fromkeys(wl):
    m = wl == k; print(f"{k:10s} | {y[m][0]:.3f} | {pred[m].mean():.3f} | {pred[m].std():.3f} | {pred[m].mean()-y[m][0]:+.3f} | {np.abs(pred[m]-y[m]).mean():.3f} | {np.abs(after[m]-y[m]).mean():.3f} | {(after[m]==0).mean():.2f}")
print(f"overall MAE {np.abs(pred-y).mean():.3f}  corr {np.corrcoef(pred, y)[0,1]:.3f}")
