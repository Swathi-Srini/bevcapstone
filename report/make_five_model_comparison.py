"""Tables and figures for the document report/Five_Model_Comparison.tex: the five models, four weather conditions, 200 episodes each.

Models: Baseline Driving Model Trained Without Traffic (repo_bc_clear), Baseline Driving Model Trained With Traffic (repo_bc_sparse), Proposed Bird's-Eye-View Driving Model (ov_split),
        Caution-Study Caution-Aware Driving Model (caut_auto), Image-Only Driving Model (vcam; real camera image with the teammates' weather filter).
Sources: bev_rl/adaptive/results/n200_all_weather/<arm>__<weather>.json  and  bev_rl/adaptive/results/vanilla_camera/vcam__<cond>__n200_s5000.json
Writes report/tables/five_model_all_metrics.tex, report/tables/five_model_paired_vs_image_only.tex and three figures:
    fig_five_model_success_by_weather, fig_five_model_outcome_mix_by_weather, fig_five_model_energy_and_speed
Cells that are not yet complete (image-only runs in progress) are shown as "not yet evaluated" and are left out of the figures.
Run from the repository root:   venv\\Scripts\\python.exe report\\make_five_model_comparison.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as F   # noqa: E402
from make_caution_comparison import mcnemar   # noqa: E402

import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.patches import Patch   # noqa: E402

ALLW = F.RES / "n200_all_weather"
VCAM = F.RES / "vanilla_camera"
CELLS = [("clear_0.0", "clear", "Clear"), ("fog_1.0", "fog1", "Fog (intensity 1.0)"), ("rain_0.6", "rain0.6", "Rain (intensity 0.6)"), ("rain_1.0", "rain1", "Rain (intensity 1.0)")]
MODELS = [("repo_bc_clear", "Baseline Driving Model Trained Without Traffic", F.ORANGE), ("repo_bc_sparse", "Baseline Driving Model Trained With Traffic", "#f2a27f"),
          ("ov_split", "Proposed Bird's-Eye-View Driving Model", F.BLUE), ("caut_auto", "Caution-Study Caution-Aware Driving Model", F.AQUA),
          ("vcam", "Image-Only Driving Model (front camera)", F.MAGENTA)]


def load(arm, cell):
    cw, vc, _ = cell
    f = (VCAM / f"vcam__{vc}__n200_s5000.json") if arm == "vcam" else (ALLW / f"{arm}__{cw}.json")
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if len(d["episodes"]) != 200:
        return None
    d["wh_per_km"] = float(np.mean([e["wh_per_km"] for e in d["episodes"]]))
    d["route"] = float(np.mean([e["route"] for e in d["episodes"]]))
    d["mean_speed"] = float(np.mean([e["speed"] for e in d["episodes"]]))
    return d


def main():
    data = {(m, c[0]): load(m, c) for m, _, _ in MODELS for c in CELLS}
    lines = ["\\begin{tabular}{@{}llrrrrrrrr@{}}\\toprule",
             "Weather & Model & Success & 95\\% interval & Collision & Left road & Timeout & Speed (m/s) & Route completion & Energy (Wh per km)\\\\\\midrule"]
    for c in CELLS:
        for i, (m, label, _) in enumerate(MODELS):
            d = data[(m, c[0])]
            wl = c[2] if i == 0 else ""
            if d is None:
                lines.append(f"{wl} & {label} & \\multicolumn{{8}}{{l}}{{not yet evaluated}}\\\\")
                continue
            lo, hi = F.wilson(round(d["success"] * 200), 200)
            lines.append(f"{wl} & {label} & {d['success']:.3f} & [{lo:.3f}, {hi:.3f}] & {d['crash']:.3f} & {d['off_road']:.3f} & {d['timeout']:.3f} & {d['mean_speed']:.2f} & {d['route']:.3f} & {d['wh_per_km']:.1f}\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule\\end{tabular}"
    (F.TAB / "five_model_all_metrics.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # paired: each other model against the image-only model, per weather
    pl = ["\\begin{tabular}{@{}llrrrr@{}}\\toprule", "Weather & Model & Difference in success (model minus image-only) & Only this model succeeded & Only image-only succeeded & Exact McNemar $p$\\\\\\midrule"]
    for c in CELLS:
        v = data[("vcam", c[0])]
        if v is None:
            continue
        sv = {e["seed"]: e["outcome"] == "success" for e in v["episodes"]}
        first = True
        for m, label, _ in MODELS[:-1]:
            d = data[(m, c[0])]
            if d is None:
                continue
            sd = {e["seed"]: e["outcome"] == "success" for e in d["episodes"]}
            x, y, p = mcnemar(sd, sv)
            pl.append(f"{c[2] if first else ''} & {label} & {(sum(sd.values()) - sum(sv.values())) / 200:+.3f} & {x} & {y} & {F.pfmt(p)}\\\\")
            first = False
        pl.append("\\midrule")
    pl[-1] = "\\bottomrule\\end{tabular}"
    (F.TAB / "five_model_paired_vs_image_only.tex").write_text("\n".join(pl) + "\n", encoding="utf-8")

    # ---- number macros for the image-only text of the document
    nm = []
    names = {"clear_0.0": "Clear", "fog_1.0": "Fog", "rain_0.6": "RainMid", "rain_1.0": "RainHigh"}
    for c in CELLS:
        v = data[("vcam", c[0])]
        k = names[c[0]]
        if v is None:
            nm += [f"\\newcommand{{\\VcamSuccess{k}}}{{(running)}}", f"\\newcommand{{\\VcamLow{k}}}{{--}}", f"\\newcommand{{\\VcamHigh{k}}}{{--}}"]
        else:
            lo, hi = F.wilson(round(v["success"] * 200), 200)
            nm += [f"\\newcommand{{\\VcamSuccess{k}}}{{{v['success']:.3f}}}", f"\\newcommand{{\\VcamLow{k}}}{{{lo:.3f}}}", f"\\newcommand{{\\VcamHigh{k}}}{{{hi:.3f}}}"]
    done = [c for c in CELLS[1:] if data[("vcam", c[0])] is not None]
    if len(done) < 3:
        sent = "The three weather runs of the image-only model had not all finished when this document was built, so its weather results are not stated here."
    else:
        parts = []
        for c in CELLS[1:]:
            v = data[("vcam", c[0])]
            parts.append(f"{c[2].lower()}: success {v['success']:.3f}, left the road {v['off_road']:.3f}, collision {v['crash']:.3f}, timeout {v['timeout']:.3f}, route completion {v['route']:.2f}, speed {v['mean_speed']:.2f}\\,m/s, energy {v['wh_per_km']:.1f}\\,Wh per km")
        cl = data[("vcam", "clear_0.0")]
        sent = "In weather (the same weather filter on the camera picture): " + "; ".join(parts) + f". For comparison, clear: success {cl['success']:.3f}, left the road {cl['off_road']:.3f}, route completion {cl['route']:.2f}."
    nm.append("\\newcommand{\\VcamWeatherSentence}{" + sent + "}")
    (F.TAB / "five_model_numbers.tex").write_text("\n".join(nm) + "\n", encoding="utf-8")

    xs = np.arange(len(CELLS))
    w = 0.16
    xlab = "Weather condition (0 = none, 1 = maximum intensity)"
    # ---- figure 1: success
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for mi, (m, label, col) in enumerate(MODELS):
        for ci, c in enumerate(CELLS):
            d = data[(m, c[0])]
            if d is None:
                continue
            xx = xs[ci] + (mi - 2) * w
            ax.bar(xx, d["success"], width=w * 0.92, color=col)
            lo, hi = F.wilson(round(d["success"] * 200), 200)
            ax.plot([xx, xx], [lo, hi], color=F.INK, lw=1.0)
            ax.text(xx, hi + 0.012, f"{d['success']:.2f}", ha="center", fontsize=6.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[2] for c in CELLS], fontsize=8)
    ax.set_xlabel(xlab, fontsize=8)
    ax.set_ylabel("Success rate (fraction of 200 episodes)", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, color=F.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend([Patch(color=col) for _, _, col in MODELS], [l for _, l, _ in MODELS], frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2,
              title="Model (bars left to right in each group; black line = 95% interval)", title_fontsize=7.4)
    ax.set_title("Success rate of the five models in four weather conditions (200 episodes each)", fontsize=9, loc="left")
    F.save(fig, "fig_five_model_success_by_weather")
    # ---- figure 2: outcome mix
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 3.6), sharex=True)
    outs = [("success", "Reached the destination", F.GREEN), ("crash", "Collision", F.RED), ("off_road", "Left the road", F.YELLOW), ("timeout", "Ran out of time", "#9a9893")]
    for ax, c in zip(axes, CELLS):
        for mi, (m, label, col) in enumerate(MODELS):
            d = data[(m, c[0])]
            if d is None:
                continue
            left = 0.0
            for key, _, oc in outs:
                ax.barh(mi, d[key], left=left, color=oc, height=0.72)
                left += d[key]
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels([f"{i + 1}" for i in range(len(MODELS))] if ax is not axes[0] else [f"{i + 1}" for i in range(len(MODELS))], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(c[2], fontsize=8, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of the 200 episodes", fontsize=7.4)
    axes[0].set_ylabel("Model number (see legend)", fontsize=8)
    handles = [Patch(color=oc) for _, _, oc in outs] + [Patch(color="none")] * 0
    leg1 = fig.legend(handles, [l for _, l, _ in outs], loc="lower left", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.0, -0.06), title="How an episode ended", title_fontsize=7.4)
    fig.legend([Patch(color=col) for _, _, col in MODELS], [f"{i + 1} = {l}" for i, (_, l, _) in enumerate(MODELS)], loc="upper left", ncol=2, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.0, -0.17), title="Models (colour of the number's model as in the other figures)", title_fontsize=7.2)
    fig.add_artist(leg1)
    fig.suptitle("How the episodes of the five models ended, in each weather condition", fontsize=9, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    F.save(fig, "fig_five_model_outcome_mix_by_weather")
    # ---- figure 3: energy and speed
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.2), gridspec_kw={"hspace": 0.5})
    for ax, key, yl, title in zip(axes, ("wh_per_km", "mean_speed"), ("Estimated traction energy (watt-hours per kilometre)", "Mean driving speed (metres per second)"), ("Energy use", "Speed")):
        for mi, (m, label, col) in enumerate(MODELS):
            for ci, c in enumerate(CELLS):
                d = data[(m, c[0])]
                if d is None:
                    continue
                xx = xs[ci] + (mi - 2) * w
                ax.bar(xx, d[key], width=w * 0.92, color=col)
                ax.text(xx, d[key] * 1.01, f"{d[key]:.0f}" if key == "wh_per_km" else f"{d[key]:.1f}", ha="center", fontsize=6.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([c[2] for c in CELLS], fontsize=8)
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_ylabel(yl, fontsize=7.6)
        ax.yaxis.grid(True, color=F.GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9, loc="left")
    axes[1].legend([Patch(color=col) for _, _, col in MODELS], [l for _, l, _ in MODELS], frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=2,
                   title="Model (bars left to right in each group)", title_fontsize=7.4)
    fig.suptitle("Estimated energy and speed of the five models in four weather conditions", fontsize=9, x=0.01, ha="left", y=0.97)
    F.save(fig, "fig_five_model_energy_and_speed")
    print({k: (None if v is None else round(v["success"], 3)) for k, v in data.items()})


if __name__ == "__main__":
    main()
