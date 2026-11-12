"""Fit the measured depth/detection degradation (measure_depth_error.py) into (1) a proof table + figure, (2) a sampling model for the policy-side environment.

Outputs
  bev_rl/adaptive/results/depth_error_model.json   : per condition and per range bin: p_detect, p_valid, rel_err samples  (consumed by degraded_env.py)
  paper/depth_degradation/tables/depth_table.{md,csv}, figures/depth_degradation.png : the evidence that weather degrades the perception the policy depends on
    python -m bev_rl.adaptive.fit_depth_error_model
Definitions (all against MetaDrive truth): truth_m = camera-forward distance to the visible rear surface; rel_err = (pipe_m - truth_m)/truth_m for detections whose box has valid stereo depth.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
_ap = argparse.ArgumentParser()                      # overridable paths (used only by the synthetic self-test; the pipeline uses the defaults)
_ap.add_argument("--src", default="bev_rl/adaptive/results/depth_error_measurements.json"); _ap.add_argument("--dst", default="bev_rl/adaptive/results/depth_error_model.json")
_ap.add_argument("--out", default="paper/depth_degradation")
_a, _ = _ap.parse_known_args()
SRC, DST, OUT = Path(_a.src), Path(_a.dst), Path(_a.out); (OUT / "tables").mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(parents=True, exist_ok=True)
BINS = [0.0, 6.0, 12.0, 30.0]
MAX_SAMPLES = 400
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6})


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def main():
    D = json.loads(SRC.read_text()); rows = D["rows"]
    keys = []
    for w, l in D["conds"]:
        keys.append(f"{w}:{l}")
    model = {"range_bins_m": BINS, "cells": {}, "source": {"file": str(SRC), "seeds": D["seeds"], "density": D["density"], "n_rows": len(rows)}}
    table = []
    for w, l in D["conds"]:
        k = f"{w}:{l}"; R = [r for r in rows if r["weather"] == w and abs(r["level"] - l) < 1e-9]
        if not R: continue
        truth = np.array([r["truth_m"] for r in R]); det = np.array([r["detected"] for r in R], bool)
        sg = np.array([np.nan if r["sgbm_m"] is None else r["sgbm_m"] for r in R]); pm = np.array([np.nan if r["pipe_m"] is None else r["pipe_m"] for r in R])
        valid = ~np.isnan(pm)
        rel_all = (pm[valid] - truth[valid]) / truth[valid]
        cell = {"n": [], "p_detect": [], "p_valid": [], "rel_err": [], "pooled_rel_err": rel_all[:MAX_SAMPLES].tolist() if len(rel_all) else [0.0]}
        pool_pd = float(det.mean()); pool_pv = float(valid[det].mean()) if det.any() else 0.0; PRIOR = 5.0     # shrink thin range bins toward the cell's pooled rate (5 pseudo-observations)
        for b0, b1 in zip(BINS[:-1], BINS[1:]):
            m = (truth >= b0) & (truth < b1); n = int(m.sum()); nd = int((m & det).sum()); nv = int((m & det & valid).sum())
            cell["n"].append(n); cell["p_detect"].append((nd + PRIOR * pool_pd) / (n + PRIOR)); cell["p_valid"].append((nv + PRIOR * pool_pv) / (nd + PRIOR))
            e = ((pm[m & det & valid] - truth[m & det & valid]) / truth[m & det & valid])
            cell["rel_err"].append(e[:MAX_SAMPLES].tolist() if len(e) >= 5 else None)
        model["cells"][k] = cell
        n = len(R); sgv = ~np.isnan(sg); ae = np.abs(sg[sgv] - truth[sgv])
        lo, hi = wilson(int(det.sum()), n)
        table.append([k, n, f"{det.mean():.3f} [{lo:.2f}, {hi:.2f}]", f"{sgv.mean():.3f}",
                      f"{np.median(ae):.2f}" if len(ae) else "n/a", f"{np.percentile(ae, 90):.2f}" if len(ae) else "n/a",
                      f"{(np.abs(rel_all) > 0.2).mean():.3f}" if len(rel_all) else "n/a", f"{valid.sum() / max(1, det.sum()):.3f}"])
    DST.write_text(json.dumps(model))
    header = ["condition", "n (vehicle-frames)", "YOLO detection rate [95% CI]", "stereo depth valid at truth pixel", "median |depth err| m", "90th pct |depth err| m", "share of detections with |rel err|>20%", "detections with valid box depth"]
    with open(OUT / "tables" / "depth_table.csv", "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows([header] + table)
    (OUT / "tables" / "depth_table.md").write_text("**Measured degradation of the GitHub stereo-depth + YOLO perception under the GitHub weather augmentation (vs MetaDrive truth)**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in table) + "\n", encoding="utf-8")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8)); order = [t[0] for t in table]; x = np.arange(len(order))
    axs[0].bar(x, [float(t[2].split()[0]) for t in table], color="#2a78d6"); axs[0].set_ylabel("YOLO detection rate"); axs[0].set_ylim(0, 1.05)
    axs[1].bar(x, [float(t[3]) for t in table], color="#1baf7a"); axs[1].set_ylabel("stereo depth valid at truth pixel"); axs[1].set_ylim(0, 1.05)
    axs[2].bar(x, [float(t[4]) if t[4] != "n/a" else 0 for t in table], color="#eb6834"); axs[2].set_ylabel("median |depth error| (m)")
    for ax in axs: ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "depth_degradation.png"); plt.close(fig)
    print((OUT / "tables" / "depth_table.md").read_text(encoding="utf-8")); print("model ->", DST)


if __name__ == "__main__":
    main()
