"""n=40 FULL story (density 0.05, seeds 1010-1049, horizon 800): the six presented models x four weather conditions, every metric.
Models: repo_bc_sparse, repo_bc_clear, ov_split, ctrl_nocaut, caut_c0, caut_auto.
Read-only on bev_rl/adaptive/results/** and bev_rl/adaptive/runs/*/log.json. Writes paper/n40_story/figures/*.png and tables/*.{md,csv}.
Run from repo root:  venv/Scripts/python.exe paper/make_n40_full_story.py

TWO HARNESSES (disclosed on every relevant figure/table):
  A  results/sparse_traffic_weather_robustness_<arm>_d0.05.json : repo_bc_sparse, repo_bc_clear, ov_split (records outcome, speed, route; NO energy)
  B  results/caution/<arm>.json                                 : ctrl_nocaut ('ctrl'), caut_c0, caut_auto, and ov_split ('base_ovsplit') (also records Wh/km)
ov_split is the same checkpoint in both; the two harnesses sample detector dropout differently, so they differ by one episode at rain 0.6.
Success/crash/speed/route use each model's own harness (ov_split -> A); Wh/km for ov_split uses B."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import binomtest

R = Path("bev_rl/adaptive/results"); RUNS = Path("bev_rl/adaptive/runs"); OUT = Path("paper/n40_story"); F = OUT / "figures"; T = OUT / "tables"
F.mkdir(parents=True, exist_ok=True); T.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "axes.edgecolor": "#6b6b66", "text.color": "#2b2b29",
                     "axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
MODELS = ["repo_bc_sparse", "repo_bc_clear", "ov_split", "ctrl_nocaut", "caut_c0", "caut_auto"]
COL = {"repo_bc_sparse": "#eb6834", "repo_bc_clear": "#1baf7a", "ov_split": "#2a78d6", "ctrl_nocaut": "#c9a227", "caut_c0": "#7a5c3e", "caut_auto": "#c2185b", "base_a": "#9a9a94"}
NICE = {"repo_bc_sparse": "Repo BC (sparse traffic)", "repo_bc_clear": "Repo BC (clear, no traffic)", "ov_split": "ov_split (ours)", "ctrl_nocaut": "ctrl_nocaut (round-0 BC)",
        "caut_c0": "caut_c0 (caution off)", "caut_auto": "caut_auto (caution = weather)", "base_a": "DAgger, GitHub BEV"}
WX = ["clear:0.0", "fog:1.0", "rain:0.6", "rain:1.0"]; WLAB = {"clear:0.0": "clear", "fog:1.0": "fog 1.0", "rain:0.6": "rain 0.6", "rain:1.0": "rain 1.0"}
OC = {"success": "#d5d7cf", "crash": "#c8372d", "off_road": "#e8a33d", "timeout": "#5c6bc0"}
HARNESS = {"repo_bc_sparse": "A", "repo_bc_clear": "A", "ov_split": "A", "ctrl_nocaut": "B", "caut_c0": "B", "caut_auto": "B"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def table(name, header, rows, caption):
    with open(T / f"{name}.csv", "w", newline="") as f: csv.writer(f).writerows([header] + rows)
    (T / f"{name}.md").write_text(f"**{caption}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n", encoding="utf-8")
    print("table", name)


def save(fig, name, rect=None): fig.tight_layout(rect=rect); fig.savefig(F / f"{name}.png"); plt.close(fig); print("fig", name)


def jl(p): return json.loads(Path(p).read_text())


def load():
    D = {m: {w: {} for w in WX} for m in MODELS}; E = {w: {} for w in WX}
    for m in ("repo_bc_sparse", "repo_bc_clear", "ov_split"):
        for e in jl(R / f"sparse_traffic_weather_robustness_{m}_d0.05.json")["episodes"]:
            D[m][e["weather"]][e["seed"]] = dict(outcome=e["outcome"], speed=e["speed"], route=e["route"], wh=None)
    for m, f in (("ctrl_nocaut", "ctrl"), ("caut_c0", "caut_c0"), ("caut_auto", "caut_auto")):
        for e in jl(R / "caution" / f"{f}.json")["episodes"]:
            D[m][e["weather"]][e["seed"]] = dict(outcome=e["outcome"], speed=e["mean_speed"], route=e["route_completion"], wh=e["wh_per_km"])
    for e in jl(R / "caution" / "base_ovsplit.json")["episodes"]:
        E[e["weather"]][e["seed"]] = dict(outcome=e["outcome"], speed=e["mean_speed"], route=e["route_completion"], wh=e["wh_per_km"])
    return D, E


def stat(eps):
    v = list(eps.values()); n = len(v); oc = [x["outcome"] for x in v]; k = oc.count("success"); lo, hi = wilson(k, n)
    wh = [x["wh"] for x in v if x["wh"] is not None]
    return dict(n=n, k=k, p=k / n, lo=lo, hi=hi, crash=oc.count("crash") / n, off=oc.count("off_road") / n, to=oc.count("timeout") / n,
                speed=float(np.mean([x["speed"] for x in v])), route=float(np.mean([x["route"] for x in v])), wh=float(np.mean(wh)) if wh else None)


def legend_top(fig, models, y=1.02):
    fig.legend(handles=[Patch(color=COL[m], label=NICE[m]) for m in models], ncol=3, loc="upper center", bbox_to_anchor=(0.5, y), frameon=False, fontsize=8.5)


# ---------------------------------------------------------------------------------------------------------------------------
def fig_success(D, S):
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, w in zip(axs, WX):
        for i, m in enumerate(MODELS):
            s = S[m][w]; ax.bar(i, s["p"], 0.7, color=COL[m]); ax.errorbar(i, s["p"], yerr=[[s["p"] - s["lo"]], [s["hi"] - s["p"]]], color="#2b2b29", lw=1, capsize=2.5)
            ax.text(i, min(s["hi"] + 0.02, 1.07), f"{s['p']:.3f}", ha="center", fontsize=7)
        ax.set_xticks([]); ax.set_ylim(0, 1.12); ax.set_title(WLAB[w], fontsize=10, loc="left"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("success rate, n=40 (95% Wilson CI)"); legend_top(fig, MODELS, 0.995)
    fig.text(0.01, 0.005, "density 0.05, seeds 1010-1049, horizon 800. Weather = detector dropout only (keep-prob clear 1.00 / fog 0.27 / rain0.6 0.16 / rain1.0 0.04).", fontsize=7.5, color="#55554f")
    save(fig, "n40_fig1_success_by_weather", rect=(0, 0.04, 1, 0.88))


def fig_outcomes(S):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, w in zip(axs, WX):
        for i, m in enumerate(MODELS):
            s = S[m][w]; left = 0
            for lab, v, c in (("success", s["p"], OC["success"]), ("crash", s["crash"], OC["crash"]), ("off_road", s["off"], OC["off_road"]), ("timeout", s["to"], OC["timeout"])):
                if v > 0:
                    ax.barh(i, v, left=left, color=c, edgecolor="white", lw=1.2)
                    if v >= 0.12 or lab == "success": ax.text(left + v / 2, i, f"{v:.3f}" if lab == "success" else f"{v:.2f}", ha="center", va="center", fontsize=7, color="#2b2b29" if lab == "success" else "white")
                left += v
        ax.set_xlim(0, 1); ax.set_title(WLAB[w], fontsize=10, loc="left"); ax.grid(axis="y", visible=False); ax.set_xlabel("share of 40 episodes")
    axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels([NICE[m] for m in MODELS], fontsize=8); axs[0].invert_yaxis()
    fig.legend(handles=[Patch(color=OC[k], label=k.replace("_", "-")) for k in OC], ncol=4, loc="upper center", bbox_to_anchor=(0.55, 0.995), frameon=False)
    save(fig, "n40_fig2_outcome_breakdown", rect=(0, 0, 1, 0.9))


def fig_efficiency(D, E, S):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0)); xs = np.arange(4)
    for ax, (key, lab) in zip(axs, (("speed", "mean speed (m/s)"), ("wh", "energy (Wh/km)"), ("route", "route completion"))):
        for j, m in enumerate(MODELS):
            if key == "wh":
                src = E if m == "ov_split" else {w: D[m][w] for w in WX}
                vals = [np.mean([x["wh"] for x in src[w].values() if x["wh"] is not None]) if any(x["wh"] is not None for x in src[w].values()) else np.nan for w in WX]
            else:
                vals = [S[m][w][key] for w in WX]
            if np.all(np.isnan(vals)): continue
            ax.plot(xs + (j - 2.5) * 0.06, vals, "-o", color=COL[m], lw=1.6, ms=4.5, label=NICE[m])
        ax.set_xticks(xs); ax.set_xticklabels([f"{WLAB[w]}\nkeep {k}" for w, k in zip(WX, ("1.00", "0.27", "0.16", "0.04"))], fontsize=8); ax.set_ylabel(lab)
    fig.text(0.01, 0.005, "Repo models: Wh/km not recorded at n=40 (n=200 clear: 86 and 91 Wh/km). ov_split energy comes from the caution harness (same checkpoint). Seeds 1010-1049, density 0.05.", fontsize=7.5, color="#55554f")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)
    save(fig, "n40_fig3_speed_energy_route_by_weather", rect=(0, 0.04, 1, 0.86))


def fig_tradeoff(D, E):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in ("ctrl_nocaut", "ov_split", "caut_c0", "caut_auto"):
        src = E if m == "ov_split" else D[m]; X = [np.mean([x["speed"] for x in src[w].values()]) for w in WX]; Y = [np.mean([x["wh"] for x in src[w].values()]) for w in WX]
        ax.plot(X, Y, "-o", color=COL[m], lw=1.4, ms=6, mfc="white" if m != "caut_auto" else COL[m], label=NICE[m])
        if m == "caut_auto":
            for x, y, w in zip(X, Y, WX): ax.annotate(WLAB[w], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("mean speed (m/s)"); ax.set_ylabel("energy (Wh/km)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Only caut_auto moves: slower and cheaper in fog/rain, but with timeouts (0.075 / 0.10 / 0.15)", loc="left", fontsize=9)
    save(fig, "n40_fig4_speed_vs_energy_tradeoff")


def fig_seed_heatmap(D):
    seeds = sorted(D["ov_split"]["clear:0.0"]); code = {"success": 0, "crash": 1, "off_road": 2, "timeout": 3}
    fig, axs = plt.subplots(4, 1, figsize=(12.5, 8.2), sharex=True)
    for ax, w in zip(axs, WX):
        M = np.array([[code[D[m][w][s]["outcome"]] for s in seeds] for m in MODELS])
        ax.pcolormesh(M, cmap=ListedColormap([OC["success"], OC["crash"], OC["off_road"], OC["timeout"]]), vmin=0, vmax=3, edgecolors="white", linewidth=0.6)
        ax.set_yticks(np.arange(len(MODELS)) + 0.5); ax.set_yticklabels([NICE[m] for m in MODELS], fontsize=7.5); ax.invert_yaxis(); ax.grid(False)
        ax.set_title(WLAB[w], fontsize=9, loc="left")
    axs[-1].set_xticks(np.arange(len(seeds)) + 0.5); axs[-1].set_xticklabels(seeds, rotation=90, fontsize=7); axs[-1].set_xlabel("scenario seed (identical for every model)")
    fig.legend(handles=[Patch(color=OC[k], label=k.replace("_", "-")) for k in OC], ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(F / "n40_fig5_per_seed_outcomes.png"); plt.close(fig); print("fig n40_fig5_per_seed_outcomes")


def fig_n40_vs_n200(S):
    files = {"repo_bc_sparse": "repo_bc_sparse", "repo_bc_clear": "repo_bc_clear", "ov_split": "base_ovsplit", "ctrl_nocaut": "ctrl_nocaut"}; n200 = {}
    for m, f in files.items():
        p = R / "final_200seed" / f"{f}.json"
        if p.exists(): oc = [e["outcome"] for e in jl(p)["episodes"]]; k = oc.count("success"); n200[m] = (k / len(oc),) + wilson(k, len(oc))
    fig, ax = plt.subplots(figsize=(8.2, 4.4)); order = MODELS[::-1]
    for i, m in enumerate(order):
        s = S[m]["clear:0.0"]; ax.plot([s["lo"], s["hi"]], [i + 0.13] * 2, color=COL[m], lw=1.4, alpha=0.6); ax.plot(s["p"], i + 0.13, "o", mfc="white", mec=COL[m], mew=1.8, ms=8)
        if m in n200:
            p, lo, hi = n200[m]; ax.plot([lo, hi], [i - 0.13] * 2, color=COL[m], lw=1.8); ax.plot(p, i - 0.13, "o", color=COL[m], ms=8); ax.text(hi + 0.012, i - 0.13, f"{p:.3f}", va="center", fontsize=8)
        ax.text(s["hi"] + 0.012, i + 0.13, f"{s['p']:.3f}", va="center", fontsize=8, color="#55554f")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[m] for m in order], fontsize=8.5); ax.set_xlim(0.55, 1.08); ax.set_xlabel("success rate (95% Wilson CI), clear weather, density 0.05")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker="o", mfc="white", mec="#2b2b29", ls="", label="n=40 (seeds 1010-1049)"), plt.Line2D([], [], marker="o", color="#2b2b29", ls="", label="n=200 (fresh seeds 5000-5199)")],
              frameon=False, loc="lower left", fontsize=8)
    ax.set_title("ctrl_nocaut drops 0.975 -> 0.895 on fresh seeds; others stay within their intervals", loc="left", fontsize=9)
    fig.text(0.01, 0.005, "caut_c0 / caut_auto were not run at n=200.", fontsize=7.5, color="#55554f")
    save(fig, "n40_fig6_n40_vs_n200_clear", rect=(0, 0.03, 1, 1))


def loc_lines(arm, cell_re=None):
    j = jl(R / f"localisation_error_sensitivity_{arm}_d0.05.json"); c = j["cells"]
    return [(v["lat_m"], v["success"], v["n"]) for v in sorted(c.values(), key=lambda v: v["lat_m"])]


def fig_localisation():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    def draw(ax, series, title):
        for lab, col, pts, ls in series:
            x = [p[0] for p in pts]; y = [p[1] for p in pts]; ci = [wilson(round(p[1] * p[2]), p[2]) for p in pts]
            ax.fill_between(x, [c[0] for c in ci], [c[1] for c in ci], color=col, alpha=0.12, lw=0); ax.plot(x, y, ls, color=col, lw=1.8, marker="o", ms=5, label=lab)
        ax.set_xlabel("map-pose lateral error (m)"); ax.set_title(title, fontsize=9.5, loc="left"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    draw(axs[0], [("Repo BC (sparse)", COL["repo_bc_sparse"], loc_lines("repo_bc_sparse"), "-"), ("DAgger, GitHub BEV (no overlay)", COL["base_a"], loc_lines("base_a"), "-"), ("ov_split (ours)", COL["ov_split"], loc_lines("ov_split"), "-")],
         "Models vs. localisation error (n=40/level)")
    draw(axs[1], [("ov_split", COL["ov_split"], loc_lines("ov_split"), "-"), ("pose_aug unconditioned", "#6d8f3a", loc_lines("pose_aug_unconditioned"), "--"), ("pose_aug conditioned", "#8a4fbf", loc_lines("pose_aug_conditioned"), "--")],
         "Pose-augmented variants: no gain over ov_split")
    two = {}
    for arm in ("pose_cam_unconditioned", "pose_cam_conditioned"):
        for v in jl(R / f"pose_two_cue_{arm}_d0.05.json")["cells"].values(): two.setdefault((arm, v["cam_level"]), []).append((v["lat_m"], v["success"], v["n"]))
    ser = [("ov_split (map only)", COL["ov_split"], loc_lines("ov_split"), "-")]
    for (arm, cam), pts in sorted(two.items()):
        ser.append((f"{arm.replace('pose_cam_', 'two-cue ')} (cam noise {cam})", "#8a4fbf" if "cond" in arm and "uncond" not in arm else "#6d8f3a", sorted(pts), "-" if cam > 0.5 else ":"))
    draw(axs[2], ser, "Two-cue variants (extra lateral/heading cue)")
    save(fig, "n40_fig7_localisation_error")


def fig_ctrl_rounds(S):
    pts = [("round 0\n(chosen, plain BC)", S["ctrl_nocaut"]["clear:0.0"]["p"], S["ctrl_nocaut"]["clear:0.0"]["lo"], S["ctrl_nocaut"]["clear:0.0"]["hi"])]
    for it in (2, 5):
        p = R / f"checkpoint_eval_ctrl_nocaut_iter{it}.json"
        if p.exists(): j = jl(p); pts.append((f"round {it}", j["success"], j["success_ci"][0], j["success_ci"][1]))
    s = S["ov_split"]["clear:0.0"]; pts.append(("ov_split\n(for reference)", s["p"], s["lo"], s["hi"]))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, (lab, p, lo, hi) in enumerate(pts):
        ax.bar(i, p, 0.55, color=COL["ov_split"] if "ov_split" in lab else COL["ctrl_nocaut"]); ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="#2b2b29", capsize=4); ax.text(i, hi + 0.02, f"{p:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.895, color="#c8372d", ls="--", lw=1.3); ax.text(-0.4, 1.16, "dashed line = ctrl_nocaut round 0 on 200 fresh seeds (0.895)", ha="left", fontsize=8, color="#c8372d")
    ax.set_xticks(range(len(pts))); ax.set_xticklabels([p[0] for p in pts], fontsize=8.5); ax.set_ylim(0, 1.25); ax.set_ylabel("success, clear, n=40 (95% CI)"); ax.grid(axis="x", visible=False)
    ax.set_title("ctrl_nocaut by DAgger checkpoint (seeds 1010-1049)", loc="left", fontsize=9.5); save(fig, "n40_fig8_ctrl_nocaut_by_checkpoint")


def fig_ppo():
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); names = {"rl_plain": "plain reward", "rl_head": "head only", "rl_full": "full + energy", "rl_energy": "energy penalty", "rl_safe": "safety-shaped"}
    cols = ["#6d8f3a", "#8a4fbf", "#eb6834", "#2a78d6", "#c2185b"]
    for (r, lab), c in zip(names.items(), cols):
        p = RUNS / r / "log.json"
        if not p.exists(): continue
        L = jl(p); ax.plot([x["steps"] / 1000 for x in L], [x["val_success"] for x in L], "-o", color=c, lw=1.4, ms=4, label=lab)
    ax.axhspan(0.35 - 0.075, 0.35 + 0.075, color="#d5d7cf", alpha=0.5, lw=0); ax.text(150, 0.435, "starting score 0.35 +- 1 s.e. (40 episodes)", ha="right", fontsize=7.5, color="#55554f")
    ax.set_xlabel("PPO training steps (thousands)"); ax.set_ylabel("validation success (40 mixed-traffic episodes)"); ax.set_ylim(0.2, 0.55); ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    ax.set_title("PPO fine-tuning of ov_split: every variant stays inside the noise band", loc="left", fontsize=9.5); save(fig, "n40_fig9_ppo_negative_result")


def fig_caution(D, S):
    kp = {k: v["keep_prob"] for k, v in jl(R / "sparse_traffic_weather_robustness_ov_split_d0.05.json")["cells"].items()}; c = [1 - kp[w] for w in WX]
    fig, axs = plt.subplots(1, 3, figsize=(12.2, 3.8))
    for m in ("caut_c0", "caut_auto", "ctrl_nocaut"):
        axs[0].plot(c, [S[m][w]["speed"] for w in WX], "-o", color=COL[m], label=NICE[m]); axs[1].plot(c, [S[m][w]["to"] for w in WX], "-o", color=COL[m]); axs[2].plot(c, [S[m][w]["p"] for w in WX], "-o", color=COL[m])
    for ax, lab in zip(axs, ("mean speed (m/s)", "timeout rate", "success rate")): ax.set_xlabel("caution c = 1 - detector keep-prob"); ax.set_ylabel(lab)
    axs[2].set_ylim(0, 1.05); axs[1].set_ylim(-0.01, 0.2); axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    for ax, w in zip([axs[0]], WX[:0]): pass
    for x, w in zip(c, WX): axs[0].annotate(WLAB[w], (x, S["caut_auto"][w]["speed"]), textcoords="offset points", xytext=(7, 6), fontsize=7.5, color="#55554f")
    axs[0].text(0.99, 0.55, "ctrl_nocaut and caut_c0 overlap\n(identical speed)", transform=axs[0].transAxes, ha="right", fontsize=7.5, color="#55554f")
    fig.suptitle("Caution mechanism: behaviour changes (slower), success does not improve", x=0.01, ha="left", fontsize=10); fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "n40_fig10_caution_effect.png"); plt.close(fig); print("fig n40_fig10_caution_effect")


# --------------------------------------------------------------------------------------------- tables
def tables(D, E, S):
    rows = []
    for m in MODELS:
        for w in WX:
            s = S[m][w]
            wh = f"{(np.mean([x['wh'] for x in E[w].values()]) if m == 'ov_split' else s['wh']):.1f}" if (m == "ov_split" or s["wh"] is not None) else "not recorded"
            rows.append([NICE[m], HARNESS[m], WLAB[w], s["n"], f"{s['p']:.3f}", f"[{s['lo']:.2f}, {s['hi']:.2f}]", f"{s['crash']:.3f}", f"{s['off']:.3f}", f"{s['to']:.3f}", f"{s['speed']:.2f}", wh, f"{s['route']:.3f}"])
    table("n40_table1_full_results", ["model", "harness", "weather", "n", "success", "95% CI", "crash", "off-road", "timeout", "speed m/s", "Wh/km", "route compl."], rows,
          "All six models x four weather conditions, density 0.05, seeds 1010-1049. Harness A=robustness script (no energy), B=caution script. ov_split Wh/km from B.")
    prow = []
    for w in WX:
        seeds = sorted(D["ov_split"][w]); a = np.array([D["ov_split"][w][s]["outcome"] == "success" for s in seeds])
        for m in MODELS:
            if m == "ov_split": continue
            o = np.array([D[m][w][s]["outcome"] == "success" for s in seeds]); b = int((a & ~o).sum()); c = int((~a & o).sum()); p = binomtest(b, b + c, .5).pvalue if b + c else 1.0
            prow.append([WLAB[w], f"ov_split - {m}", f"{a.mean() - o.mean():+.3f}", b, c, f"{p:.3f}"])
    table("n40_table2_paired_vs_ov_split", ["weather", "comparison", "success diff", "ov_split only wins", "other only wins", "exact McNemar p"], prow,
          "Paired comparison on identical seeds (n=40). With n=40 only large gaps can reach p<0.05.")
    srow = []
    for m in MODELS:
        ps = [S[m][w]["p"] for w in WX]; fs = lambda w: {s for s, e in D[m][w].items() if e["outcome"] != "success"}; same = sum(fs(w) == fs("clear:0.0") for w in WX[1:])
        srow.append([NICE[m], *[f"{p:.3f}" for p in ps], f"{max(ps) - min(ps):.3f}", f"{same}/3"])
    table("n40_table3_weather_sensitivity", ["model", "clear", "fog 1.0", "rain 0.6", "rain 1.0", "range (max-min)", "weathers whose failed-seed set equals clear's"], srow,
          "Weather sensitivity. Identical failed-seed sets across weathers mean the failures are scenario-driven, not weather-driven.")
    files = {"repo_bc_sparse": "repo_bc_sparse", "repo_bc_clear": "repo_bc_clear", "ov_split": "base_ovsplit", "ctrl_nocaut": "ctrl_nocaut"}; rrow = []
    for m in MODELS:
        s = S[m]["clear:0.0"]; p = R / "final_200seed" / f"{files.get(m, 'x')}.json"
        if p.exists(): oc = [e["outcome"] for e in jl(p)["episodes"]]; k = oc.count("success"); lo, hi = wilson(k, len(oc)); r200 = f"{k/len(oc):.3f} [{lo:.2f}, {hi:.2f}]"; d = f"{k/len(oc) - s['p']:+.3f}"
        else: r200, d = "not run", "-"
        rrow.append([NICE[m], f"{s['p']:.3f} [{s['lo']:.2f}, {s['hi']:.2f}]", r200, d])
    table("n40_table4_n40_vs_n200", ["model", "n=40 clear (1010-1049)", "n=200 clear (5000-5199)", "change"], rrow, "Replication on fresh seeds. The 0.975 of ctrl_nocaut does not replicate.")
    seeds = sorted(D["ov_split"]["clear:0.0"]); sh = {"success": "S", "crash": "C", "off_road": "O", "timeout": "T"}; note = {1019: "vehicle pulls out of a side lane", 1039: "oncoming vehicle sideswipes ego (ego centred)", 1040: "roundabout merge"}; frow = []
    for s in seeds:
        o = [D[m]["clear:0.0"][s]["outcome"] for m in MODELS]
        if any(x != "success" for x in o): frow.append([s, *[sh[x] for x in o], note.get(s, "")])
    table("n40_table5_failure_seeds_clear", ["seed", *[NICE[m] for m in MODELS], "cause (from earlier trace)"], frow, "Seeds where at least one model fails (clear). S success, C crash, O off-road, T timeout.")
    lrow = []; arms = [("repo_bc_sparse", "repo_bc_sparse"), ("base_a", "base_a"), ("ov_split", "ov_split"), ("pose_aug_unconditioned", "pose_aug_unconditioned"), ("pose_aug_conditioned", "pose_aug_conditioned")]
    for nm, a in arms:
        pts = loc_lines(a); lrow.append([NICE.get(nm, nm), *[f"{p[1]:.3f}" for p in pts]])
    table("n40_table6_localisation", ["model", "0 m", "0.25 m", "0.5 m", "1.0 m", "2.0 m"], lrow, "Success vs. map-pose lateral error (heading error scales with it), n=40 per level, clear, density 0.05. Injection mechanisms differ between model families.")


def main():
    D, E = load()
    for m in MODELS:
        for w in WX: assert len(D[m][w]) == 40, (m, w, len(D[m][w]))
    S = {m: {w: stat(D[m][w]) for w in WX} for m in MODELS}
    for f in (lambda: fig_success(D, S), lambda: fig_outcomes(S), lambda: fig_efficiency(D, E, S), lambda: fig_tradeoff(D, E), lambda: fig_seed_heatmap(D),
              lambda: fig_n40_vs_n200(S), fig_localisation, lambda: fig_ctrl_rounds(S), fig_ppo, lambda: fig_caution(D, S)):
        try: f()
        except Exception as e: print("FIGURE FAILED:", repr(e)); plt.close("all")
    tables(D, E, S)


if __name__ == "__main__":
    main()
