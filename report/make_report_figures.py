"""Build every figure of the two reports from the saved result files (nothing is typed in by hand except the two bird's-eye-view fidelity series,
which are copied from report/data/birds_eye_view_fidelity_diagnosis.md and marked as such).

Run from the repository root:
    venv\\Scripts\\python.exe report\\make_report_figures.py

Output: report/figures/*.pdf and *.png (the vector PDF is what the LaTeX documents include), report/tables/paired_headline_versus_others.tex and
        report/tables/weather_calibrated_perception.tex

Every figure has: a descriptive title, labelled axes with units, a legend that names every model (or outcome) shown, and consistent colours:
    blue = the proposed model and its variants, orange = the two baseline models supplied by the teammates, grey = rule-based teachers that see the exact simulator state,
    teal = the image-only model, violet / pink = other learned models; green / red / yellow / grey in outcome charts always mean success / collision / left the road / timeout.
The plain-English interpretation and the research implication of each figure are written in the LaTeX captions (report/figure_blocks/).
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
RES = ROOT / "bev_rl" / "adaptive" / "results"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
GREY = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd7"
OUTCOME_COL = {"success": "#1baf7a", "crash": "#e34948", "off_road": "#eda100", "timeout": "#9a9992"}
OUTCOME_LABEL = {"success": "Success (reached the destination)", "crash": "Collision", "off_road": "Left the road", "timeout": "Timeout (horizon reached)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

NAME = {   # internal code name -> full descriptive name used in every figure and table
    "base_ovsplit": "Proposed Bird's-Eye-View Driving Model\n(map overlay + 3 vehicle-state inputs)",
    "ov_split": "Proposed Bird's-Eye-View Driving Model\n(map overlay + 3 vehicle-state inputs)",
    "ctrl_nocaut": "Caution-Study Control Model\n(caution input fixed to zero)",
    "repo_bc_clear": "Baseline Driving Model\nTrained Without Traffic",
    "repo_bc_sparse": "Baseline Driving Model\nTrained With Traffic",
    "stock_idm": "Stock Intelligent Driver Model\n(rule-based; sees exact simulator state)",
    "safe_teacher": "Collision-Aware Teacher\n(rule-based; sees exact simulator state)",
    "deg_base": "Weather-Degradation-Trained\nDriving Model",
    "ppo_plain": "Reinforcement-Learning-Fine-Tuned Model\n(attempt 4, no weather input)",
    "image_only": "Image-Only Driving Model\n(reads front-camera pixels)",
}
ROLE = {"base_ovsplit": BLUE, "ov_split": BLUE, "ctrl_nocaut": "#7fb0ea", "deg_base": VIOLET, "ppo_plain": MAGENTA, "repo_bc_clear": ORANGE, "repo_bc_sparse": ORANGE,
        "stock_idm": GREY, "safe_teacher": GREY, "image_only": AQUA}
ROLE_LEGEND = [("Proposed model", BLUE), ("Caution-study control (variant of the proposed model)", "#7fb0ea"), ("Baseline models supplied by the teammates", ORANGE),
               ("Rule-based teachers (see the exact simulator state)", GREY), ("Image-only model", AQUA)]


def flat(name):
    return name.replace("\n", " ")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=170)
    plt.close(fig)
    print("figure", name)


def load_final(arm):
    return json.loads((RES / "final_200seed" / f"{arm}.json").read_text())


def load_image_only():
    f = RES / "vanilla_camera" / "vcam__clear__n200_s5000.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    d["wh_per_km"] = float(np.mean([e["wh_per_km"] for e in d["episodes"]]))
    d["n"] = len(d["episodes"])
    return d


ARMS = ["safe_teacher", "base_ovsplit", "ctrl_nocaut", "stock_idm", "repo_bc_clear", "repo_bc_sparse"]
FINAL = {a: load_final(a) for a in ARMS}
_image_only = load_image_only()
if _image_only is not None and _image_only["n"] == 200:
    FINAL["image_only"] = _image_only
    ARMS = ARMS + ["image_only"]


def role_legend(ax, loc="lower right", arms=None):
    present = set(ROLE[a] for a in (arms or ARMS))
    handles = [Patch(color=c, label=l) for l, c in ROLE_LEGEND if c in present]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, frameon=False, fontsize=7.4, title="Colour = kind of model", title_fontsize=8)


# ---- 1. headline success ------------------------------------------------------------------------------------------------------
def fig_headline():
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    order = ARMS[::-1]
    for i, a in enumerate(order):
        d = FINAL[a]
        k = round(d["success"] * d["n"])
        lo, hi = wilson(k, d["n"])
        ax.barh(i, d["success"], color=ROLE[a], height=0.62)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
        ax.text(max(hi, d["success"]) + 0.012, i, f"{d['success']:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=8.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[a] for a in order], fontsize=7.8)
    ax.set_xlim(0.0, 1.28)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Success rate (fraction of the 200 test episodes that reached the destination)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Success rate of the driving models on the same 200 test episodes\n(clear weather, sparse traffic; bar = success rate, black line = 95% confidence interval)", fontsize=9, loc="left")
    role_legend(ax, "lower right")
    save(fig, "fig_headline_success_clear_weather_200_episodes")


# ---- 2. outcome breakdown -----------------------------------------------------------------------------------------------------
def fig_outcomes():
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    order = ARMS[::-1]
    for i, a in enumerate(order):
        d = FINAL[a]
        left = 0
        for key in ("success", "crash", "off_road", "timeout"):
            v = d[key]
            ax.barh(i, v, left=left, color=OUTCOME_COL[key], height=0.62, edgecolor="white", linewidth=1.2)
            if v >= 0.045:
                ax.text(left + v / 2, i, f"{v:.3f}", ha="center", va="center", color="white" if key in ("success", "crash") else INK, fontsize=7.6)
            left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[a] for a in order], fontsize=7.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of the 200 test episodes (fraction; each bar sums to 1)")
    ax.legend(handles=[Patch(color=OUTCOME_COL[k], label=OUTCOME_LABEL[k]) for k in OUTCOME_COL], ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.04), frameon=False, fontsize=8)
    ax.set_title("How the 200 test episodes end, for each model", fontsize=9, loc="left", pad=36)
    save(fig, "fig_outcome_breakdown_200_episodes")


# ---- 3. per-seed strips ---------------------------------------------------------------------------------------------------------
def fig_strips():
    seeds = sorted(e["seed"] for e in FINAL["base_ovsplit"]["episodes"])
    order = ["base_ovsplit", "safe_teacher", "ctrl_nocaut", "stock_idm", "repo_bc_clear", "repo_bc_sparse"] + (["image_only"] if "image_only" in FINAL else [])
    out = {a: {e["seed"]: e["outcome"] for e in FINAL[a]["episodes"]} for a in order}
    nfail = {s: sum(out[a][s] != "success" for a in order) for s in seeds}
    seeds_sorted = sorted(seeds, key=lambda s: (nfail[s], s))
    mat = np.array([[0 if out[a][s] == "success" else 1 for s in seeds_sorted] for a in order])
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.imshow(mat, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#d7efe6", "#e34948"]), interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([flat(NAME[a]).split("(")[0].strip() for a in order], fontsize=7.6)
    ax.set_xlabel("The 200 test episodes, sorted from 'every model succeeds' (left) to 'most models fail' (right)")
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Which test episodes each model fails", fontsize=9, loc="left", pad=22)
    ax.legend(handles=[Patch(color="#d7efe6", label="Success"), Patch(color="#e34948", label="Failure (collision, left the road or timeout)")], ncol=2, loc="lower left",
              bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8)
    save(fig, "fig_per_episode_outcome_strips")


# ---- 4. paired difference --------------------------------------------------------------------------------------------------------
def pfmt(p):
    return "$<0.0001$" if p < 0.0001 else f"{p:.4f}"


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def fig_paired():
    ref = "base_ovsplit"
    others = ["safe_teacher", "ctrl_nocaut", "stock_idm", "repo_bc_clear", "repo_bc_sparse"] + (["image_only"] if "image_only" in FINAL else [])
    o = {a: {e["seed"]: e["outcome"] == "success" for e in FINAL[a]["episodes"]} for a in ARMS}
    seeds = sorted(o[ref])
    rng = np.random.default_rng(0)
    rows = []
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for i, a in enumerate(others[::-1]):
        d = np.array([int(o[ref][s]) - int(o[a][s]) for s in seeds])
        diff = d.mean()
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        b, c = int((d == 1).sum()), int((d == -1).sum())
        p = mcnemar_exact(b, c)
        ax.plot([lo, hi], [i, i], color=INK, lw=1.4)
        ax.plot(diff, i, "o", color=ROLE[a], ms=8, mec="white", mew=1.4)
        ax.text(max(hi, diff) + 0.012, i + 0.2, f"{diff:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=8, ha="left")
        rows.append((a, diff, lo, hi, b, c, p))
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([flat(NAME[a]).split("(")[0].strip() for a in others[::-1]], fontsize=7.8)
    ax.set_xlim(-0.12, 0.62 if "image_only" in FINAL else 0.34)
    ax.set_xlabel("Success rate of the proposed model minus success rate of the compared model (same 200 episodes; fraction)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Paired difference in success: proposed model versus each other model\n(point = difference, line = 95% bootstrap interval; right of the zero line = proposed model better)", fontsize=8.6, loc="left")
    ax.xaxis.label.set_fontsize(7.6)
    role_legend(ax, "lower right", arms=others)
    save(fig, "fig_paired_difference_headline_versus_others")
    with open(TAB / "paired_headline_versus_others.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\nCompared model & Difference & 95\\% interval & Only proposed model succeeded & Only compared model succeeded & Exact McNemar $p$ \\\\\\midrule\n")
        for a, diff, lo, hi, b, c, p in rows[::-1]:
            fh.write(f"{flat(NAME[a]).split('(')[0].strip()} & {diff:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {b} & {c} & {pfmt(p)} \\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


# ---- 5. weather under calibrated perception ----------------------------------------------------------------------------------------
def fig_weather():
    cells = ["clear_0.0", "fog_1.0", "rain_0.6", "rain_1.0"]
    labels = ["Clear", "Fog\n(intensity 1.0)", "Rain\n(intensity 0.6)", "Rain\n(intensity 1.0)"]
    models = ["ov_split", "deg_base", "ppo_plain", "repo_bc_sparse"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(cells))
    w = 0.19
    rows = []
    for m_i, m in enumerate(models):
        for c_i, c in enumerate(cells):
            d = json.loads((RES / "degraded_eval" / "cal_d0.05" / f"{m}__{c}.json").read_text())
            n = d["n"]
            lo, hi = wilson(round(d["success"] * n), n)
            xx = x[c_i] + (m_i - 1.5) * w
            ax.bar(xx, d["success"], width=w * 0.92, color=ROLE[m])
            ax.plot([xx, xx], [lo, hi], color=INK, lw=1.1)
            ax.text(xx, hi + 0.012, f"{d['success']:.2f}", ha="center", fontsize=7)
            rows.append((m, c, n, d["success"], lo, hi, d["crash"], d["off_road"], d["timeout"], d["wh_per_km"], d["mean_speed"]))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Weather condition (simulated image degradation; intensity from 0 = none to 1 = strongest)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Success rate (fraction of 200 episodes)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend([Patch(color=ROLE[m]) for m in models], [flat(NAME[m]) for m in models], ncol=2, frameon=False, fontsize=7.4, loc="lower center", bbox_to_anchor=(0.5, 1.02), title="Model (bars left to right in each group)", title_fontsize=8)
    ax.set_title("Success rate in four weather conditions under the measured perception degradation", fontsize=9, loc="left", pad=52)
    save(fig, "fig_success_by_weather_calibrated_perception_200_episodes")
    with open(TAB / "weather_calibrated_perception.tex", "w", encoding="utf-8") as fh:
        fh.write("\\begin{tabular}{llrrrrrr}\\toprule\nModel & Weather & Success & 95\\% interval & Collision & Left road & Timeout & Energy (Wh per km) \\\\\\midrule\n")
        for m, c, n, s, lo, hi, cr, off, to, wh, sp in rows:
            fh.write(f"{flat(NAME[m]).split('(')[0].strip()} & {c.replace('_', ' ')} & {s:.3f} & [{lo:.3f}, {hi:.3f}] & {cr:.3f} & {off:.3f} & {to:.3f} & {wh:.1f} \\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


# ---- 6. measured perception degradation --------------------------------------------------------------------------------------------
def fig_perception():
    rows = list(csv.DictReader(open(ROOT / "report" / "data" / "measured_perception_degradation_table.csv", encoding="utf-8")))
    conds = [r["condition"] for r in rows]
    det = [float(r["YOLO detection rate [95% CI]"].split()[0]) for r in rows]
    ci = [tuple(float(v) for v in r["YOLO detection rate [95% CI]"].split("[")[1].rstrip("]").split(",")) for r in rows]
    valid = [float(r["stereo depth valid at truth pixel"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"wspace": 0.28})
    x = np.arange(len(conds))
    pretty = [c.replace("none:0.0", "clear").replace(":", "\n") for c in conds]
    pretty = [p.replace("fog\n", "fog\n").replace("rain\n", "rain\n") for p in pretty]
    for ax, vals, col, title, ylab in ((axes[0], det, BLUE, "Share of vehicles found by the object detector", "Fraction of vehicles detected"),
                                      (axes[1], valid, AQUA, "Share of vehicle pixels with a valid stereo depth", "Fraction of pixels with valid depth")):
        ax.bar(x, vals, color=col, width=0.62)
        for i, v in enumerate(vals):
            if ax is axes[0]:
                ax.plot([i, i], list(ci[i]), color=INK, lw=1.1)
                ax.text(i, ci[i][1] + 0.02, f"{v:.2f}", ha="center", fontsize=7.2)
            else:
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.2)
        ax.set_xticks(x)
        ax.set_xticklabels(pretty, fontsize=6.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylab)
        ax.set_xlabel("Weather condition (type, intensity 0-1)")
        ax.set_title(title, fontsize=8.4, loc="left")
    fig.suptitle("Measured effect of simulated fog and rain on the camera perception (40 scenes; black lines = 95% intervals)", fontsize=9, x=0.01, ha="left", y=1.02)
    save(fig, "fig_measured_perception_degradation")


# ---- 7. imitation training validation curves -----------------------------------------------------------------------------------------
def fig_dagger():
    runs = [("ov_split", "Proposed model: map overlay + 3 vehicle-state inputs", BLUE), ("base_a", "Plain bird's-eye view + 6 vehicle-state inputs", ORANGE),
            ("ov_both", "Map overlay + 6 vehicle-state inputs", VIOLET), ("ov_bev_only", "Map overlay, no vehicle-state inputs", AQUA),
            ("arch_scalar_only", "No bird's-eye view, 6 vehicle-state inputs only", MAGENTA)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, key, title in ((axes[0], "d0.05", "Sparse traffic (density 0.05)"), (axes[1], "d0.2", "Dense traffic (density 0.2)")):
        for run, label, col in runs:
            log = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / run / "log.json").read_text())
            ax.plot([r["iter"] for r in log], [r["val"][key]["success"] for r in log], "-o", color=col, lw=1.8, ms=4.5, label=label)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Training round (0 = teacher drives; later rounds = model drives)")
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Validation success rate\n(fraction of 10 validation episodes)")
    axes[0].legend(frameon=False, fontsize=7.4, loc="upper center", bbox_to_anchor=(1.04, -0.26), ncol=2, title="Input design of the model", title_fontsize=8)
    fig.suptitle("Validation success after each training round, for five input designs", fontsize=9, x=0.01, ha="left", y=1.0)
    save(fig, "fig_dataset_aggregation_validation_curves")


# ---- 8. localisation error ---------------------------------------------------------------------------------------------------------
def fig_localisation():
    rows = list(csv.DictReader(open(ROOT / "report" / "data" / "localisation_error_sensitivity_table.csv", encoding="utf-8")))
    groups = {}
    for r in rows:
        groups.setdefault(r["policy"], []).append(r)
    style = {"DAgger, map-overlay BEV (ov_split)": ("Proposed Bird's-Eye-View Driving Model", BLUE), "DAgger, GitHub BEV (base_a)": ("Plain bird's-eye view + 6 vehicle-state inputs (imitation-trained)", VIOLET),
             "Repo BC (sparse traffic)": ("Baseline Driving Model Trained With Traffic", ORANGE)}
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for pol, rs in groups.items():
        label, col = style[pol]
        xs = [float(r["lateral error (m)"]) for r in rs]
        ys = [float(r["success"]) for r in rs]
        lo = [float(r["success 95% CI"].strip("[]").split(",")[0]) for r in rs]
        hi = [float(r["success 95% CI"].strip("[]").split(",")[1]) for r in rs]
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=5, label=label)
    ax.set_xlabel("Persistent lateral error of the car's position against the map (metres; heading error grows with it, about 0.1 radian per metre)", fontsize=7.6)
    ax.set_ylabel("Success rate (fraction of 40 episodes)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left", title="Model (band = 95% confidence interval)", title_fontsize=8)
    ax.set_title("Success when the car's position estimate is wrong by a constant amount", fontsize=9, loc="left")
    save(fig, "fig_success_versus_localisation_error")


def fig_pose_training():
    def curve(fname, cam=None):
        d = json.loads((RES / fname).read_text())
        pts = {}
        for e in d["episodes"]:
            if cam is not None and abs(e.get("cam_level", -1) - cam) > 1e-6:
                continue
            pts.setdefault(e["lat_m"], []).append(e["outcome"] == "success")
        xs = sorted(pts)
        return xs, [float(np.mean(pts[x])) for x in xs], [wilson(int(sum(pts[x])), len(pts[x])) for x in xs]
    series = [("localisation_error_sensitivity_ov_split_d0.05.json", None, "Proposed model (trained without position error)", BLUE, "-"),
              ("localisation_error_sensitivity_pose_aug_unconditioned_d0.05.json", None, "Trained with random position error; map is the only lane cue", AQUA, "--"),
              ("localisation_error_sensitivity_pose_aug_conditioned_d0.05.json", None, "Same, and told the size of the position error", VIOLET, "--"),
              ("pose_two_cue_pose_cam_unconditioned_d0.05.json", 0.33, "Trained with position error + a second noisy lane cue", GREEN, "-"),
              ("pose_two_cue_pose_cam_conditioned_d0.05.json", 0.33, "Same, and told the size of both errors", MAGENTA, "-")]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for fname, cam, label, col, ls in series:
        xs, ys, cis = curve(fname, cam)
        ax.plot(xs, ys, ls, color=col, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(xs, [c[0] for c in cis], [c[1] for c in cis], color=col, alpha=0.10, lw=0)
    ax.set_xlabel("Persistent lateral error of the car's position against the map (metres; heading error about 0.1 radian per metre)", fontsize=7.8)
    ax.set_ylabel("Success rate (fraction of 40 episodes)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.3, loc="lower left", title="Model (dashed = map is the only lane cue; solid = has a second cue)", title_fontsize=7.8)
    ax.set_title("Training with position error, and adding a second lane cue", fontsize=9, loc="left")
    save(fig, "fig_success_versus_localisation_error_pose_trained_models")


# ---- 10. teacher ceilings -----------------------------------------------------------------------------------------------------------
def fig_teacher():
    rows = list(csv.DictReader(open(ROOT / "report" / "data" / "teacher_ceiling_dense_traffic_table.csv", encoding="utf-8")))
    nice = {"stock IDM": "Stock intelligent\ndriver model", "IDM, no lane change": "Driver model,\nno lane change", "IDM, no lane change, larger gap": "Driver model, no lane\nchange, larger gap", "IDM + predictive safety layer": "Driver model +\npredictive braking"}
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        left = 0
        for key in ("success", "crash", "timeout"):
            v = float(r[key])
            ax.bar(i, v, bottom=left, color=OUTCOME_COL[key], width=0.6, edgecolor="white", linewidth=1.2)
            if v > 0.04:
                ax.text(i, left + v / 2, f"{v:.2f}", ha="center", va="center", color="white" if key != "timeout" else INK, fontsize=8)
            left += v
    ax.set_xticks(x)
    ax.set_xticklabels([nice.get(r["teacher"], r["teacher"]) for r in rows], fontsize=7.8)
    ax.set_xlabel("Rule-based teacher variant (all see the exact simulator state)")
    ax.set_ylabel("Share of 20 episodes (fraction)")
    ax.legend(handles=[Patch(color=OUTCOME_COL[k], label=OUTCOME_LABEL[k]) for k in ("success", "crash", "timeout")], frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title("Even the rule-based teachers fail most dense-traffic episodes (traffic density 0.2)", fontsize=9, loc="left")
    save(fig, "fig_teacher_ceiling_dense_traffic")


# ---- 11. speed versus energy ---------------------------------------------------------------------------------------------------------
def fig_energy():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    off = {"safe_teacher": (8, -4), "base_ovsplit": (8, -22), "ctrl_nocaut": (8, 6), "stock_idm": (8, -14), "repo_bc_clear": (8, 6), "repo_bc_sparse": (8, 6), "image_only": (8, 6)}
    for a in ARMS:
        d = FINAL[a]
        ax.plot(d["mean_speed"], d["wh_per_km"], "o", color=ROLE[a], ms=9, mec="white", mew=1.4)
        ax.annotate(NAME[a].split("\n")[0], (d["mean_speed"], d["wh_per_km"]), xytext=off[a], textcoords="offset points", fontsize=7.2)
    ax.set_xlabel("Mean driving speed (metres per second)")
    ax.set_ylabel("Estimated traction energy (watt-hours per kilometre)")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    allx = [FINAL[a]["mean_speed"] for a in ARMS]
    ally = [FINAL[a]["wh_per_km"] for a in ARMS]
    ax.set_xlim(min(allx) - 0.25, max(allx) + 0.85)
    ax.set_ylim(min(ally) - 4, max(ally) + 6)
    ax.set_title("Speed and estimated energy use of each model (200 clear-weather episodes)", fontsize=9, loc="left")
    role_legend(ax, "upper right")
    save(fig, "fig_speed_versus_energy_200_episodes")


# ---- 12. bird's-eye-view fidelity (numbers copied from report/data/birds_eye_view_fidelity_diagnosis.md) -------------------------------------
def fig_bev_fidelity():
    steps_a, true_a, bev_a = [308, 309, 313, 316, 317], [14.70, 13.25, 7.30, 3.01, 1.56], [np.nan, 11.32, 8.12, 3.64, 2.21]
    steps_b, true_b, bev_b = [70, 71, 72, 78, 80, 81], [9.3, 8.2, 7.15, 2.67, 1.09, 0.49], [np.nan, np.nan, 8.03, 4.38, 2.81, np.nan]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"wspace": 0.28})
    for ax, st, tr, bv, title in ((axes[0], steps_a, true_a, bev_a, "Seed 1039: an oncoming vehicle drifts\ninto the lane (closing speed 15 m/s)"),
                                   (axes[1], steps_b, true_b, bev_b, "Seed 1040: a vehicle crosses from the left\n(roundabout merge)")):
        ax.plot(st, tr, "-o", color=INK, lw=1.8, ms=5, label="True gap (from the simulator)")
        ax.plot(st, bv, "-s", color=RED, lw=1.8, ms=5, label="Gap drawn in the bird's-eye view")
        absent = [(s_, t_) for s_, t_, b_ in zip(st, tr, bv) if np.isnan(b_)]
        if absent:
            ax.plot([a[0] for a in absent], [a[1] for a in absent], "o", ms=13, mfc="none", mec=RED, mew=1.6, label="Vehicle missing from the view")
        ax.set_title(title, fontsize=8.4, loc="left")
        ax.set_xlabel("Simulation step (10 steps = 1 second)")
        ax.yaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    axes[0].set_ylabel("Distance to the nearest vehicle (metres)")
    fig.suptitle("Gap to the other vehicle as drawn in the model's input, in the last steps before two collisions", fontsize=9, x=0.01, ha="left", y=1.02)
    save(fig, "fig_birds_eye_view_fidelity_before_collision")


# ---- 13. reinforcement learning: training curves of the four attempts ---------------------------------------------------------------------
def fig_rl_training():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    ax = axes[0, 0]
    for run, label, col in (("C2", "Run C2 (1.5 million steps)", BLUE), ("probe_C", "Run Probe C (1 million steps)", ORANGE)):
        d = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / run / "train_log.json").read_text())
        ax.plot([r["steps"] / 1e6 for r in d], [r["success"] for r in d], "-o", ms=3.5, color=col, label=label)
    ax.set_title("Attempt 1: privileged teacher", fontsize=8.6, loc="left")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Success in training episodes\n(fraction)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    d = json.loads((ROOT / "archive" / "removed_scripts_and_early_runs_2026-11-05" / "runs" / "pilot_cond" / "log.json").read_text())
    ax.plot([r["steps"] / 1e3 for r in d], [r["val_success"] for r in d], "-o", color=VIOLET, label="Fine-tuned proposed model\n(with weather input)")
    ax.set_title("Attempt 2: fine-tuning the proposed model", fontsize=8.6, loc="left")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Validation success (fraction\nof 40 validation episodes)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1, 0]
    runs = [("rl_plain", "Base reward", BLUE), ("rl_energy", "+ energy term", ORANGE), ("rl_safe", "+ safe speed, scaled crash penalty", AQUA), ("rl_head", "+ headway term", VIOLET), ("rl_full", "all terms", MAGENTA)]
    for run, label, col in runs:
        log = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / run / "log.json").read_text())
        ax.plot([r["steps"] / 1e3 for r in log], [r["val_success"] for r in log], "-o", ms=3.5, lw=1.5, color=col, label=label)
    ax.set_title("Attempt 3: five reward combinations", fontsize=8.6, loc="left")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Validation success (fraction of\n40 validation episodes, dense traffic)")
    ax.set_ylim(0.2, 0.55)
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    ax = axes[1, 1]
    log = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / "ppo_plain" / "log.json").read_text())
    ax.plot([r["steps"] / 1e3 for r in log], [r["val_success"] for r in log], "-o", color=MAGENTA, lw=2, label="Fine-tuned from the weather-degradation-trained model")
    ax.set_title("Attempt 4: reward version 3", fontsize=8.6, loc="left")
    ax.set_xlabel("Training steps (thousands)")
    ax.set_ylabel("Validation success (fraction of\n12 validation episodes)")
    ax.set_ylim(0, 1.3)
    ax.legend(frameon=False, fontsize=6.4, loc="upper center")
    fig.suptitle("What the training logs of the four reinforcement-learning attempts show", fontsize=9.5, x=0.01, ha="left", y=0.97)
    save(fig, "fig_reinforcement_learning_training_curves")


# ---- 14. reinforcement learning: start versus end ----------------------------------------------------------------------------------------
def fig_rl_before_after():
    import csv as _csv

    def ep_counts(f):
        rows = list(_csv.DictReader(open(f, encoding="utf-8")))
        return sum(r["outcome"] == "success" for r in rows), len(rows)
    s_idm, n_idm = ep_counts(RES / "idm" / "episodes_idm.csv")
    s_c2, n_c2 = ep_counts(RES / "C2" / "episodes_C2.csv")
    s_pr, n_pr = ep_counts(RES / "probe" / "episodes_probe_C.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw={"hspace": 0.62, "wspace": 0.32})
    ax = axes[0, 0]
    vals = [s_idm / n_idm, s_c2 / n_c2, s_pr / n_pr]
    ax.bar(range(3), vals, color=[GREY, BLUE, ORANGE], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Stock driver\nmodel (what the\nteacher copied)", "Run C2", "Run Probe C"], fontsize=6.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success on 70 test\nepisodes (fraction)")
    ax.set_title("Attempt 1: privileged teacher", fontsize=8.6, loc="left")
    ax = axes[0, 1]
    ax.bar(range(2), [0.675, 0.675], color=[GREY, VIOLET], width=0.5)
    for i, v in enumerate([0.675, 0.675]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Before training\n(step 0)", "After 120 000\nsteps"], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation success\n(fraction of 40 episodes)")
    ax.set_title("Attempt 2: fine-tuning the\nproposed model", fontsize=8.6, loc="left")
    ax = axes[1, 0]
    names = ["Base\nreward", "+ energy", "+ safe speed\n+ crash scale", "+ headway", "All\nterms"]
    runs = ["rl_plain", "rl_energy", "rl_safe", "rl_head", "rl_full"]
    ends = []
    for r in runs:
        log = json.loads((ROOT / "bev_rl" / "adaptive" / "runs" / r / "log.json").read_text())
        ends.append(log[-1]["val_success"])
    xs = np.arange(len(runs))
    ax.bar(xs - 0.2, [0.35] * 5, width=0.38, color=GREY, label="Before training (all five start identical)")
    ax.bar(xs + 0.2, ends, width=0.38, color=MAGENTA, label="After 150 000 steps")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=5.6)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Validation success (fraction of\n40 dense-traffic episodes)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    ax.set_title("Attempt 3: reward sweep", fontsize=8.6, loc="left")
    ax = axes[1, 1]
    cells = ["clear_0.0", "fog_1.0", "rain_0.6", "rain_1.0"]
    lab = ["Clear", "Fog 1.0", "Rain 0.6", "Rain 1.0"]
    a_ = [json.loads((RES / "degraded_eval" / "cal_d0.05" / f"deg_base__{c}.json").read_text())["success"] for c in cells]
    b_ = [json.loads((RES / "degraded_eval" / "cal_d0.05" / f"ppo_plain__{c}.json").read_text())["success"] for c in cells]
    xs = np.arange(4)
    ax.bar(xs - 0.2, a_, width=0.38, color=VIOLET, label="Before reinforcement learning")
    ax.bar(xs + 0.2, b_, width=0.38, color=MAGENTA, label="After 300 000 steps")
    ax.set_xticks(xs)
    ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Success on 200 test\nepisodes (fraction)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper center")
    ax.set_title("Attempt 4: reward version 3", fontsize=8.6, loc="left")
    fig.suptitle("Before and after reinforcement learning, for each of the four attempts (a different measurement in each panel)", fontsize=9, x=0.01, ha="left", y=0.97)
    save(fig, "fig_reinforcement_learning_before_and_after")


if __name__ == "__main__":
    fig_headline()
    fig_outcomes()
    fig_strips()
    fig_paired()
    fig_weather()
    fig_perception()
    fig_dagger()
    fig_localisation()
    fig_pose_training()
    fig_teacher()
    fig_energy()
    fig_bev_fidelity()
    fig_rl_training()
    fig_rl_before_after()
