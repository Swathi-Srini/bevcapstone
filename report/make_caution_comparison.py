"""Table and figure for the 200-episode, four-weather comparison of the caution-aware model with the proposed model and the two baseline models.

Reads bev_rl/adaptive/results/n200_all_weather/<arm>__<weather>.json (written by evaluate_n200_all_weather.py: seeds 5000-5199, traffic density 0.05,
horizon 800, weather = detector dropout at the measured retention). Writes
    report/tables/caution_comparison_200_episodes.tex      every metric for every model and weather condition
    report/tables/caution_comparison_paired_tests.tex       paired tests of the caution-aware model against each other model
    report/figures/fig_caution_comparison_200_episodes.pdf/.png
Run from the repository root:   venv\\Scripts\\python.exe report\\make_caution_comparison.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402

import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402

RES = F.RES / "n200_all_weather"
CELLS = ["clear_0.0", "fog_1.0", "rain_0.6", "rain_1.0"]
CELL_LABEL = {"clear_0.0": "Clear", "fog_1.0": "Fog (intensity 1.0)", "rain_0.6": "Rain (intensity 0.6)", "rain_1.0": "Rain (intensity 1.0)"}
MODELS = [("repo_bc_clear", "Baseline Driving Model Trained Without Traffic", F.ORANGE), ("repo_bc_sparse", "Baseline Driving Model Trained With Traffic", "#f2a27f"),
          ("ov_split", "Proposed Bird's-Eye-View Driving Model", F.BLUE), ("caut_auto", "Caution-Study Caution-Aware Driving Model (caution from weather)", F.AQUA)]


def load(arm, cell):
    f = RES / f"{arm}__{cell}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("n") == 200 and len(d["episodes"]) == 200 else None


def mcnemar(a, b):
    x = sum(a[s] and not b[s] for s in a)
    y = sum(b[s] and not a[s] for s in a)
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p


def main():
    data = {(m, c): load(m, c) for m, _, _ in MODELS for c in CELLS}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        print("cells not yet available:", missing)
    # ---- table with every metric
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}\\toprule",
             "Weather & Model & Success & 95\\% interval & Collision & Left road & Timeout & Speed (m/s) & Route completion & Energy (Wh per km)\\\\\\midrule"]
    for c in CELLS:
        first = True
        for m, label, _ in MODELS:
            d = data[(m, c)]
            wl = CELL_LABEL[c] if first else ""
            first = False
            if d is None:
                lines.append(f"{wl} & {label} & \\multicolumn{{8}}{{l}}{{not yet evaluated}}\\\\")
                continue
            lo, hi = F.wilson(round(d["success"] * 200), 200)
            lines.append(f"{wl} & {label} & {d['success']:.3f} & [{lo:.3f}, {hi:.3f}] & {d['crash']:.3f} & {d['off_road']:.3f} & {d['timeout']:.3f} & {d['mean_speed']:.2f} & {d['route']:.3f} & {d['wh_per_km']:.1f}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule\\end{tabular}"
    (F.TAB / "caution_comparison_200_episodes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # ---- paired tests: caution-aware model against each other model, per weather condition
    pl = ["\\begin{tabular}{@{}llrrrr@{}}\\toprule", "Weather & Compared model & Difference in success & Only caution-aware model succeeded & Only compared model succeeded & Exact McNemar $p$\\\\\\midrule"]
    for c in CELLS:
        a = data[("caut_auto", c)]
        if a is None:
            continue
        sa = {e["seed"]: e["outcome"] == "success" for e in a["episodes"]}
        first = True
        for m, label, _ in MODELS:
            if m == "caut_auto" or data[(m, c)] is None:
                continue
            sb = {e["seed"]: e["outcome"] == "success" for e in data[(m, c)]["episodes"]}
            x, y, p = mcnemar(sa, sb)
            diff = (sum(sa.values()) - sum(sb.values())) / 200
            pl.append(f"{CELL_LABEL[c] if first else ''} & {label} & {diff:+.3f} & {x} & {y} & {F.pfmt(p)}\\\\")
            first = False
        pl.append("\\midrule")
    pl[-1] = "\\bottomrule\\end{tabular}"
    (F.TAB / "caution_comparison_paired_tests.tex").write_text("\n".join(pl) + "\n", encoding="utf-8")
    # ---- figure: success, energy, speed per weather condition
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.2), gridspec_kw={"hspace": 0.45})
    x = np.arange(len(CELLS))
    w = 0.2
    panels = [("success", "Success rate (fraction of 200 episodes)", (0, 1.12), "Success rate"), ("wh_per_km", "Estimated traction energy (watt-hours per kilometre)", None, "Energy use"),
              ("mean_speed", "Mean driving speed (metres per second)", None, "Speed")]
    for ax, (key, ylabel, ylim, title) in zip(axes, panels):
        for mi, (m, label, col) in enumerate(MODELS):
            for ci, c in enumerate(CELLS):
                d = data[(m, c)]
                if d is None:
                    continue
                xx = x[ci] + (mi - 1.5) * w
                ax.bar(xx, d[key], width=w * 0.92, color=col)
                if key == "success":
                    lo, hi = F.wilson(round(d["success"] * 200), 200)
                    ax.plot([xx, xx], [lo, hi], color=F.INK, lw=1.0)
                    ax.text(xx, hi + 0.012, f"{d[key]:.2f}", ha="center", fontsize=6.4)
                else:
                    ax.text(xx, d[key] * 1.01, f"{d[key]:.1f}" if key == "wh_per_km" else f"{d[key]:.2f}", ha="center", fontsize=6.4)
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel("Weather condition (simulated image degradation; detector dropout at the measured rate)", fontsize=7.6)
        if ylim:
            ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color=F.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].legend([Patch(color=col) for _, _, col in MODELS], [label for _, label, _ in MODELS], ncol=1, frameon=False, fontsize=7.2, loc="lower left",
                   bbox_to_anchor=(0.0, 1.12), title="Model (bars left to right in each group)", title_fontsize=8)
    fig.suptitle("Caution-aware model against the proposed model and the two baselines: success, energy and speed in four weather conditions (200 episodes each)", fontsize=8.6, x=0.01, ha="left", y=1.06)
    F.save(fig, "fig_caution_comparison_200_episodes")


if __name__ == "__main__":
    main()
