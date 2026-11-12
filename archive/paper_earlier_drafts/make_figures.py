"""Regenerates every paper figure from measured result files. Run from the repo root:  venv/Scripts/python.exe paper/make_figures.py
Figures whose inputs do not exist yet are skipped (printed), never faked."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

R = Path("bev_rl/adaptive/results")
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"base": "#8c8c8c", "ours": "#1f77b4", "alt": "#ff7f0e"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def box(ax, x, y, w, h, t, fc="#eef3f8", ec="#34495e", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", lw=1))


def fig_arch():
    """Fig 1: original repo policy vs ours."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax in axs:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    a = axs[0]; a.set_title("(a) Original (repo BC baseline)", fontsize=10)
    box(a, .2, 4.2, 3, 1.2, "BEV 1x64x64\nunknown/free/ego/vehicle\n(NO route, NO boundary)")
    box(a, .2, 1.8, 3, 1.4, "6-D scalar\nspeed, progress, lateral,\nheading, curvature, goal")
    box(a, 4.2, 4.2, 2, 1.2, "CNN -> 128"); box(a, 4.2, 1.8, 2, 1.4, "MLP -> 32"); box(a, 7, 2.9, 2.6, 1.4, "head\n(steer, throttle)", fc="#fdebd0")
    for y in (4.8, 2.5):
        arrow(a, 3.2, y, 4.2, y)
    arrow(a, 6.2, 4.8, 7, 3.9); arrow(a, 6.2, 2.5, 7, 3.3)
    b = axs[1]; b.set_title("(b) Ours: caution-conditioned, map-overlay BEV", fontsize=10)
    box(b, .2, 4.4, 3, 1.2, "BEV 1x64x64 + HD-map overlay\nroute 0.5, boundary 0.8\n(+ detector dropout in training)", fc="#d6eaf8")
    box(b, .2, 2.7, 3, 1.1, "3-D scalar\nspeed, progress, goal\n(lat/heading/curv now in BEV)", fc="#d6eaf8")
    box(b, .2, 0.6, 3, 1.5, "caution c in [0,1]\nc_auto = 1 - measured\ndetector retention(weather)", fc="#d5f5e3")
    box(b, 4.2, 4.4, 2, 1.2, "CNN -> 128"); box(b, 4.2, 2.7, 2, 1.1, "MLP -> 32"); box(b, 4.2, 0.6, 2, 1.5, "Linear -> 16", fc="#d5f5e3")
    box(b, 7, 2.6, 2.6, 1.6, "head\n(steer, throttle)", fc="#fdebd0")
    for y in (5.0, 3.25, 1.35):
        arrow(b, 3.2, y, 4.2, y)
    arrow(b, 6.2, 5.0, 7, 3.9); arrow(b, 6.2, 3.25, 7, 3.4); arrow(b, 6.2, 1.35, 7, 2.9)
    b.text(5, 0.1, "teacher IDM: speed 30->20 km/h, gap 10->25 m as c: 0->1", ha="center", fontsize=8, style="italic")
    fig.tight_layout(); fig.savefig(OUT / "fig1_architecture.png"); plt.close(fig)


def fig_retention():
    """Fig 2: measured detector retention."""
    t = json.load(open(R / "retention_table.json")); fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for kind, c in (("fog", COL["alt"]), ("rain", COL["ours"])):
        pts = sorted((float(k.split(":")[1]), v["retention"], *v["ci"]) for k, v in t.items() if k.startswith(kind))
        x, y, lo, hi = map(np.array, zip(*pts))
        ax.plot(x, y, "o-", color=c, label=kind); ax.fill_between(x, lo, hi, color=c, alpha=.15)
    ax.axhline(1, color="k", lw=.5, ls=":"); ax.set_xlabel("weather intensity"); ax.set_ylabel("detections kept vs clear (paired)")
    ax.set_title("Measured YOLOv8 detection retention\n(unmodified GitHub weather+detector, 20 scenes)", fontsize=9)
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig2_detector_retention.png"); plt.close(fig)


def fig_prior():
    """Fig 3: earlier policies on reserved seeds, clear weather."""
    rows = {(r["policy"], r["density"]): r for r in json.load(open(R / "repo_baselines.json"))}
    data = []
    for p, l in (("bc_sparse_traffic_5eps", "repo BC"), ("IDM_expert", "IDM expert")):
        data.append((l, [rows[(p, d)]["success"] for d in (0.05, 0.2)], [rows[(p, d)]["n"] for d in (0.05, 0.2)]))
    for run, l in (("base_a", "DAgger BEV+scalar"), ("ov_both", "DAgger overlay, 6 scalars"), ("ov_split", "DAgger overlay, 3 scalars (ov_split)")):
        f = R / f"final_{run}_best.json"
        if f.exists():
            d = {x["density"]: x for x in json.load(open(f))}
            data.append((l, [d[.05]["success"], d[.2]["success"]], [d[.05]["n"], d[.2]["n"]]))
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); w = .8 / len(data)
    for i, (l, s, n) in enumerate(data):
        ci = [wilson(round(si * ni), ni) for si, ni in zip(s, n)]
        ax.bar(np.arange(2) + i * w, s, w, yerr=[[si - c[0] for si, c in zip(s, ci)], [c[1] - si for si, c in zip(s, ci)]], capsize=2, label=l)
    ax.set_xticks(np.arange(2) + .4 - w / 2); ax.set_xticklabels(["traffic 0.05", "traffic 0.2"])
    ax.set_ylabel("success (seeds 1000-1009, n=10)"); ax.set_ylim(0, 1.1); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax.set_title("Earlier policies, clear weather, n=10 per bar (Wilson 95% CI; superseded by fig8 for sparse traffic)", fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig3_prior_policies.png"); plt.close(fig)


def fig_dagger():
    """Fig 4: DAgger validation curves."""
    fig, ax = plt.subplots(figsize=(6, 3.6)); got = 0
    for run in ("base_a", "ov_both", "ov_split", "ov_bev_only", "arch_scalar_only"):
        f = Path(f"bev_rl/adaptive/runs/{run}/log.json")
        if f.exists():
            L = json.load(open(f)); ax.plot([r["iter"] for r in L], [r["val"]["d0.2"]["success"] for r in L], "o-", label=run); got += 1
    if got:
        ax.set_xlabel("DAgger iteration"); ax.set_ylabel("val success, traffic 0.2 (seeds 3100-3109, n=10)"); ax.set_ylim(0, 1); ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig.savefig(OUT / "fig4_dagger_validation.png")
    plt.close(fig)


def fig_caution():
    """Fig 5: caution evaluation (only when results exist)."""
    order = ("base_ovsplit", "ctrl", "caut_c0", "caut_c1", "caut_auto", "caut_c0_nodrop")
    have = {a: json.load(open(R / "caution" / f"{a}.json")) for a in order if (R / "caution" / f"{a}.json").exists()}
    if "caut_auto" not in have:
        print("fig5: no caut_auto eval yet - skipped"); return
    labels = {"base_ovsplit": "prev. best (ov_split)", "ctrl": "control (no caution)", "caut_c0": "ours c=0", "caut_c1": "ours c=1",
              "caut_auto": "ours c=auto", "caut_c0_nodrop": "c=0, perfect perception"}
    weathers = list(next(iter(have.values()))["cells"]); fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, (metric, yl) in enumerate((("success", "success"), ("fail", "crash + off-road"), ("wh_per_km", "energy (Wh/km)"))):
        ax = axs[k]; w = .8 / len(have)
        for i, (a, d) in enumerate(have.items()):
            v = [d["cells"][x][metric] for x in weathers]
            err = None
            if metric != "wh_per_km":
                ci = [d["cells"][x][metric + "_ci"] for x in weathers]; err = [[vi - c[0] for vi, c in zip(v, ci)], [c[1] - vi for vi, c in zip(v, ci)]]
            ax.bar(np.arange(len(weathers)) + i * w, v, w, yerr=err, capsize=2, label=labels[a], color=COL["ours"] if a == "caut_auto" else None,
                   hatch="//" if a == "caut_c0_nodrop" else None)
        ax.set_xticks(np.arange(len(weathers)) + .4 - w / 2); ax.set_xticklabels(weathers, fontsize=8); ax.set_ylabel(yl)
    axs[0].legend(fontsize=6)
    fig.suptitle("Traffic 0.2, fresh seeds, detector dropout at measured rates (Wilson 95% CI)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig5_caution_results.png"); plt.close(fig)


def fig_training():
    """Fig 6: caution vs control training curves."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5)); got = 0
    for run in ("caut_a", "ctrl_nocaut"):
        f = Path(f"bev_rl/adaptive/runs/{run}/log.json")
        if f.exists():
            L = json.load(open(f)); ax.plot([r["iter"] for r in L], [r["val_success_auto"] for r in L], "o-", label=run); got += 1
    if got:
        ax.set_xlabel("DAgger iteration"); ax.set_ylabel("val success (c_auto), density 0.05, seeds 3100-3109"); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(OUT / "fig6_caution_training.png")
    else:
        print("fig6: no training logs yet - skipped")
    plt.close(fig)


def fig_progressive():
    """Fig 7 (main RQ figure): route-following and energy vs progressive perception degradation (x = measured detector retention)."""
    t = json.load(open(R / "retention_table.json"))
    labels = {"base_ovsplit": "prev. best (ov_split)", "ctrl": "control (no caution)", "caut_c0": "ours c=0", "caut_c1": "ours c=1", "caut_auto": "ours c=auto"}
    for kind in ("rain", "fog"):
        have = {a: json.load(open(R / "caution" / f"{a}_sweep_{kind}.json")) for a in labels if (R / "caution" / f"{a}_sweep_{kind}.json").exists()}
        if not have:
            print(f"fig7 ({kind}): no sweep results yet - skipped"); continue
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
        for a, d in have.items():
            cells = d["cells"]; lv = [float(k.split(":")[1]) for k in cells]
            keep = [1.0 if k.startswith("clear") else min(1.0, t.get(k, {"retention": 1.0})["retention"]) for k in cells]
            for ax, m, yl in zip(axs, ("route_completion", "success", "wh_per_km"), ("route completion", "success", "energy (Wh/km)")):
                ax.plot(lv, [cells[k][m] for k in cells], "o-", label=labels[a], lw=2.5 if a == "caut_auto" else 1.2); ax.set_xlabel(f"{kind} intensity"); ax.set_ylabel(yl)
        axs[0].legend(fontsize=7); fig.suptitle(f"Progressive {kind} degradation, traffic 0.2 (fresh seeds)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"fig7_progressive_{kind}.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_arch, fig_retention, fig_prior, fig_dagger, fig_caution, fig_training, fig_progressive):
        f()
    print("figures:", sorted(os.listdir(OUT)))
