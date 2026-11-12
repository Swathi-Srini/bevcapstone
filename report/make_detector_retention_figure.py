"""Figure: how the detector retention was measured at different weather intensities, and what the driving model actually receives.

Source: bev_rl/adaptive/results/perception_curve.json (raw per-frame counts, written by bev_rl/adaptive/measure_paired.py) and
        bev_rl/adaptive/results/retention_table.json (the table derived from it; this table is what the weather proxy reads).
Retention = number of vehicles the YOLO detector finds in the weathered frame / number it finds in the SAME frame without weather (paired, per scene).
The driving model receives min(1, retention) as the probability that each vehicle is kept in its bird's-eye-view grid (bev_rl/adaptive/caution.py keep_prob,
linear interpolation between measured intensities).

Run from the repository root:   venv\\Scripts\\python.exe report\\make_detector_retention_figure.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402

import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

RAW = json.loads((F.RES / "perception_curve.json").read_text())["rows"]
TABLE = json.loads((F.RES / "retention_table.json").read_text())
USED = {"fog": [1.0], "rain": [0.6, 1.0]}   # intensities evaluated in the 200-episode weather tests


def scene_ratio(kind, level):
    """Per scene: mean over weather draws of (detections with weather) / (detections without weather); scenes with no clean detection are skipped."""
    clean, wet = {}, {}
    for r in RAW:
        if r["weather"] == "none":
            clean[r["seed"]] = clean.get(r["seed"], 0) + r["n_yolo"]
        elif r["weather"] == kind and abs(r["level"] - level) < 1e-9:
            wet.setdefault(r["seed"], []).append(r["n_yolo"])
    # each clean scene is counted once per weather draw in the raw rows if it repeats; normalise by number of clean rows per seed
    nclean = {}
    for r in RAW:
        if r["weather"] == "none":
            nclean[r["seed"]] = nclean.get(r["seed"], 0) + 1
    out = []
    for s, c in clean.items():
        c = c / nclean[s]
        if c > 0 and s in wet:
            out.append(np.mean(wet[s]) / c)
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), sharey=True)
    n_scenes = None
    for ax, kind, name in zip(axes, ("fog", "rain"), ("Fog", "Rain")):
        pts = sorted((float(k.split(":")[1]), v) for k, v in TABLE.items() if k.startswith(kind + ":"))
        xs = [0.0] + [p[0] for p in pts]
        measured = [1.0] + [p[1]["retention"] for p in pts]
        lo = [1.0] + [p[1]["ci"][0] for p in pts]
        hi = [1.0] + [p[1]["ci"][1] for p in pts]
        used = [1.0] + [min(1.0, p[1]["retention"]) for p in pts]
        for x, _ in pts:
            r = scene_ratio(kind, x)
            n_scenes = len(r) if n_scenes is None else max(n_scenes, len(r))
            ax.scatter([x] * len(r), r, s=7, color=F.GRID, zorder=1, linewidths=0)
        ax.fill_between(xs, lo, hi, color=F.BLUE, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, measured, "-o", color=F.BLUE, lw=1.6, ms=4, zorder=3)
        ax.plot(xs, used, "--", color=F.ORANGE, lw=1.6, zorder=4)
        for u in USED[kind]:
            ax.axvline(u, color=F.INK2, lw=0.8, ls=":", zorder=0)
            ax.text(u, 1.58, f"tested\n{u:g}", ha="center", va="top", fontsize=7, color=F.INK2)
        ax.axhline(1.0, color=F.INK, lw=0.6, zorder=0)
        ax.set_title(f"{name}", fontsize=9, loc="left")
        ax.set_xlabel(f"{name} intensity (0 = none, 1 = maximum)", fontsize=7.6)
        ax.set_ylim(-0.02, 1.62)
        ax.yaxis.grid(True, color=F.GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Detector retention (vehicles found with weather / without)", fontsize=7.4)
    handles = [Line2D([], [], color=F.GRID, marker="o", ls="", ms=4, label="One scene (mean over weather draws)"),
               Line2D([], [], color=F.BLUE, marker="o", lw=1.6, ms=4, label="Mean retention over all scenes, with 95% interval (blue band)"),
               Line2D([], [], color=F.ORANGE, ls="--", lw=1.6, label="Probability that the driving model keeps a vehicle in its grid (measured retention capped at 1)")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Detector retention measured at different weather intensities, and what the driving model receives", fontsize=9, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    F.save(fig, "fig_detector_retention_by_weather_intensity")
    print("scenes per point (max):", n_scenes)


if __name__ == "__main__":
    main()
