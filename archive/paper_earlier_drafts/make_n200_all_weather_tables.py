"""n=200 ALL-WEATHER story (density 0.05, FRESH seeds 5000-5199, horizon 800): six models x clear / fog 1.0 / rain 0.6 / rain 1.0, one harness, every metric incl. energy.
Reads bev_rl/adaptive/results/n200_all_weather/<arm>__<weather>.json (missing cells are skipped and listed). Writes paper/n200_story/{figures,tables}.
Run from repo root:  venv/Scripts/python.exe paper/make_n200_all_weather_tables.py"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import binomtest

R = Path("bev_rl/adaptive/results/n200_all_weather"); OUT = Path("paper/n200_story"); F = OUT / "figures"; T = OUT / "tables"
F.mkdir(parents=True, exist_ok=True); T.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "axes.edgecolor": "#6b6b66", "text.color": "#2b2b29",
                     "axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
MODELS = ["repo_bc_sparse", "repo_bc_clear", "ov_split", "ctrl_nocaut", "caut_c0", "caut_auto"]
COL = {"repo_bc_sparse": "#eb6834", "repo_bc_clear": "#1baf7a", "ov_split": "#2a78d6", "ctrl_nocaut": "#c9a227", "caut_c0": "#7a5c3e", "caut_auto": "#c2185b"}
NICE = {"repo_bc_sparse": "Repo BC (sparse traffic)", "repo_bc_clear": "Repo BC (clear, no traffic)", "ov_split": "ov_split (ours)", "ctrl_nocaut": "ctrl_nocaut (round-0 BC)",
        "caut_c0": "caut_c0 (caution off)", "caut_auto": "caut_auto (caution = weather)"}
WX = [("clear", 0.0), ("fog", 1.0), ("rain", 0.6), ("rain", 1.0)]; WK = [f"{a}:{b}" for a, b in WX]; WF = [f"{a}_{b}" for a, b in WX]
WLAB = {"clear:0.0": "clear", "fog:1.0": "fog 1.0", "rain:0.6": "rain 0.6", "rain:1.0": "rain 1.0"}; KEEP = {"clear:0.0": "1.00", "fog:1.0": "0.27", "rain:0.6": "0.16", "rain:1.0": "0.04"}
OC = {"success": "#d5d7cf", "crash": "#c8372d", "off_road": "#e8a33d", "timeout": "#5c6bc0"}
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def table(name, header, rows, caption):
    with open(T / f"{name}.csv", "w", newline="", encoding="utf-8") as f: csv.writer(f).writerows([header] + rows)
    (T / f"{name}.md").write_text(f"**{caption}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n", encoding="utf-8")
    print("table", name)


def save(fig, name, rect=None): fig.tight_layout(rect=rect); fig.savefig(F / f"{name}.png"); plt.close(fig); print("fig", name)


def load():
    D, missing = {m: {} for m in MODELS}, []
    for m in MODELS:
        for wk, wf in zip(WK, WF):
            p = R / f"{m}__{wf}.json"
            if p.exists(): D[m][wk] = {e["seed"]: e for e in json.loads(p.read_text())["episodes"]}
            else: missing.append(f"{m}@{wk}")
    return D, missing


def stat(eps):
    v = list(eps.values()); n = len(v); oc = [x["outcome"] for x in v]; k = oc.count("success"); lo, hi = wilson(k, n)
    h = lambda key: (float(np.mean([x[key] for x in v])), float(1.96 * np.std([x[key] for x in v], ddof=1) / np.sqrt(n)))
    return dict(n=n, k=k, p=k / n, lo=lo, hi=hi, crash=oc.count("crash") / n, off=oc.count("off_road") / n, to=oc.count("timeout") / n, speed=h("speed"), wh=h("wh_per_km"), route=h("route"))


def boot(d, B=10000):
    m = float(d.mean()); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]; return m, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def legend_top(fig, models):
    fig.legend(handles=[Patch(color=COL[m], label=NICE[m]) for m in models], ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)


def fig_success(S, D):
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True)
    for ax, wk in zip(axs, WK):
        for i, m in enumerate(MODELS):
            if wk not in S[m]: continue
            s = S[m][wk]; ax.bar(i, s["p"], 0.7, color=COL[m]); ax.errorbar(i, s["p"], yerr=[[s["p"] - s["lo"]], [s["hi"] - s["p"]]], color="#2b2b29", lw=1, capsize=2.5)
            ax.text(i, min(s["hi"] + 0.015, 1.06), f"{s['p']:.3f}", ha="center", fontsize=7)
        ax.set_xticks([]); ax.set_ylim(0, 1.12); ax.set_title(f"{WLAB[wk]}  (detector keep-prob {KEEP[wk]})", fontsize=9.5, loc="left"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("success rate, n=200 (95% Wilson CI)"); legend_top(fig, MODELS)
    fig.text(0.01, 0.005, "density 0.05, fresh seeds 5000-5199, horizon 800. Weather = detector dropout only (no weather physics).", fontsize=7.5, color="#55554f")
    save(fig, "n200_fig1_success_by_weather", rect=(0, 0.04, 1, 0.88))


def fig_outcomes(S):
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, wk in zip(axs, WK):
        for i, m in enumerate(MODELS):
            if wk not in S[m]: continue
            s = S[m][wk]; left = 0
            for lab, v in (("success", s["p"]), ("crash", s["crash"]), ("off_road", s["off"]), ("timeout", s["to"])):
                if v > 0:
                    ax.barh(i, v, left=left, color=OC[lab], edgecolor="white", lw=1.2)
                    if v >= 0.10 or lab == "success": ax.text(left + v / 2, i, f"{v:.3f}" if lab == "success" else f"{v:.2f}", ha="center", va="center", fontsize=7, color="#2b2b29" if lab == "success" else "white")
                left += v
        ax.set_xlim(0, 1); ax.set_title(WLAB[wk], fontsize=10, loc="left"); ax.grid(axis="y", visible=False); ax.set_xlabel("share of 200 episodes")
    axs[0].set_yticks(range(len(MODELS))); axs[0].set_yticklabels([NICE[m] for m in MODELS], fontsize=8); axs[0].invert_yaxis()
    fig.legend(handles=[Patch(color=OC[k], label=k.replace("_", "-")) for k in OC], ncol=4, loc="upper center", bbox_to_anchor=(0.55, 0.995), frameon=False)
    save(fig, "n200_fig2_outcome_breakdown", rect=(0, 0, 1, 0.9))


def fig_efficiency(S):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0)); xs = np.arange(4)
    for ax, (key, lab) in zip(axs, (("speed", "mean speed (m/s)"), ("wh", "energy (Wh/km)"), ("route", "route completion"))):
        for j, m in enumerate(MODELS):
            idx = [i for i, wk in enumerate(WK) if wk in S[m]]
            if not idx: continue
            ax.errorbar(xs[idx] + (j - 2.5) * 0.06, [S[m][WK[i]][key][0] for i in idx], yerr=[S[m][WK[i]][key][1] for i in idx], fmt="-o", color=COL[m], lw=1.5, ms=4, capsize=2, label=NICE[m])
        ax.set_xticks(xs); ax.set_xticklabels([f"{WLAB[w]}\nkeep {KEEP[w]}" for w in WK], fontsize=8); ax.set_ylabel(lab + " (mean, 95% CI)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)
    fig.text(0.01, 0.005, "n=200 per cell, seeds 5000-5199, density 0.05. Same harness (and energy model) for every model.", fontsize=7.5, color="#55554f")
    save(fig, "n200_fig3_speed_energy_route_by_weather", rect=(0, 0.04, 1, 0.86))


def fig_tradeoff(S):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for m in MODELS:
        idx = [wk for wk in WK if wk in S[m]]
        if not idx: continue
        X = [S[m][wk]["speed"][0] for wk in idx]; Y = [S[m][wk]["wh"][0] for wk in idx]
        ax.plot(X, Y, "-o", color=COL[m], lw=1.3, ms=5.5, mfc="white" if m != "caut_auto" else COL[m], label=NICE[m])
        if m == "caut_auto":
            for x, y, wk in zip(X, Y, idx): ax.annotate(WLAB[wk], (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("mean speed (m/s)"); ax.set_ylabel("energy (Wh/km)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Speed vs. energy, n=200 per point (4 weather conditions per model)", loc="left", fontsize=9.5); save(fig, "n200_fig4_speed_vs_energy")


def tables(S, D):
    rows = []
    for m in MODELS:
        for wk in WK:
            if wk not in S[m]: continue
            s = S[m][wk]
            rows.append([NICE[m], WLAB[wk], s["n"], f"{s['p']:.3f}", f"[{s['lo']:.3f}, {s['hi']:.3f}]", f"{s['crash']:.3f}", f"{s['off']:.3f}", f"{s['to']:.3f}",
                         f"{s['speed'][0]:.2f}", f"{s['wh'][0]:.1f}", f"{s['route'][0]:.3f}"])
    table("n200_table1_full_results", ["model", "weather", "n", "success", "95% CI", "crash", "off-road", "timeout", "speed m/s", "Wh/km", "route compl."], rows,
          "ALL six models x four weather conditions, 200 fresh seeds (5000-5199), density 0.05, one harness.")
    srow = []
    for m in MODELS:
        if not all(wk in S[m] for wk in WK): continue
        ps = [S[m][wk]["p"] for wk in WK]; srow.append([NICE[m], *[f"{p:.3f}" for p in ps], f"{max(ps) - min(ps):.3f}", f"{min(ps[1:]) - ps[0]:+.3f}"])
    table("n200_table2_weather_sensitivity", ["model", "clear", "fog 1.0", "rain 0.6", "rain 1.0", "range (max-min)", "worst weather minus clear"], srow, "Success by weather (n=200 each) and its spread.")
    prow, erow = [], []
    for wk in WK:
        if wk not in S["ov_split"]: continue
        seeds = sorted(D["ov_split"][wk]); a = np.array([D["ov_split"][wk][s]["outcome"] == "success" for s in seeds])
        for m in MODELS:
            if m == "ov_split" or wk not in D[m]: continue
            o = np.array([D[m][wk][s]["outcome"] == "success" for s in seeds]); b = int((a & ~o).sum()); c = int((~a & o).sum())
            pv = binomtest(b, b + c, .5).pvalue if b + c else 1.0; d, lo, hi = boot(a.astype(float) - o.astype(float))
            prow.append([WLAB[wk], f"ov_split - {m}", f"{d:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", b, c, f"{pv:.4f}"])
            de, elo, ehi = boot(np.array([D["ov_split"][wk][s]["wh_per_km"] - D[m][wk][s]["wh_per_km"] for s in seeds]))
            ds, slo, shi = boot(np.array([D["ov_split"][wk][s]["speed"] - D[m][wk][s]["speed"] for s in seeds]))
            erow.append([WLAB[wk], f"ov_split - {m}", f"{de:+.1f}", f"[{elo:+.1f}, {ehi:+.1f}]", f"{ds:+.2f}", f"[{slo:+.2f}, {shi:+.2f}]"])
    table("n200_table3_paired_success_vs_ov_split", ["weather", "comparison", "success diff", "95% paired bootstrap CI", "ov_split only wins", "other only wins", "exact McNemar p"], prow,
          "Paired success comparison on identical seeds (n=200 per weather).")
    table("n200_table4_paired_energy_speed_vs_ov_split", ["weather", "comparison", "Wh/km diff", "95% CI", "speed diff m/s", "95% CI"], erow,
          "Paired energy and speed differences vs ov_split (negative Wh/km = ov_split uses less).")


def main():
    D, missing = load(); S = {m: {wk: stat(D[m][wk]) for wk in D[m]} for m in MODELS}
    print("missing cells:", missing or "none")
    for f in (lambda: fig_success(S, D), lambda: fig_outcomes(S), lambda: fig_efficiency(S), lambda: fig_tradeoff(S)):
        try: f()
        except Exception as e: print("FIGURE FAILED:", repr(e)); plt.close("all")
    tables(S, D)


if __name__ == "__main__":
    main()
