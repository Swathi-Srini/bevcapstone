"""ov_split story (density 0.05 ONLY): ov_split vs the two repo models (repo_bc_sparse, repo_bc_clear). Reads measured files only; missing input => skipped and printed.
Run from repo root:  venv/Scripts/python.exe paper/make_ov_split_story_figures_and_tables.py
Outputs: paper/ov_split_story/figures/*.png , paper/ov_split_story/tables/*.{md,csv}
Colour = entity (fixed): ov_split blue, repo_bc_sparse orange, repo_bc_clear aqua, base_a grey, teachers magenta. Every mark is directly labelled and each figure has a table."""
import csv, json, itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

R = Path("bev_rl/adaptive/results"); OUT = Path("paper/ov_split_story"); F = OUT / "figures"; T = OUT / "tables"
F.mkdir(parents=True, exist_ok=True); T.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160, "axes.edgecolor": "#6b6b66", "text.color": "#2b2b29",
                     "axes.grid": True, "grid.color": "#e6e6e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
COL = {"ov_split": "#2a78d6", "repo_bc_sparse": "#eb6834", "repo_bc_clear": "#1baf7a", "base_a": "#9a9a94", "stock_idm": "#e87ba4", "safe_teacher": "#8a4fbf"}
NICE = {"ov_split": "ov_split (ours)", "repo_bc_sparse": "Repo BC, sparse-traffic", "repo_bc_clear": "Repo BC, clear/no-traffic", "base_a": "DAgger, GitHub BEV",
        "stock_idm": "IDM teacher", "safe_teacher": "Safety-layer teacher"}
MAIN = ["repo_bc_sparse", "repo_bc_clear", "ov_split"]
OC = {"success": "#d5d7cf", "crash": "#c8372d", "off_road": "#e8a33d", "timeout": "#5c6bc0"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


def table(name, header, rows, caption):
    with open(T / f"{name}.csv", "w", newline="") as f: csv.writer(f).writerows([header] + rows)
    (T / f"{name}.md").write_text(f"**{caption}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n")
    print("table", name)


def save(fig, name): fig.tight_layout(); fig.savefig(F / f"{name}.png"); plt.close(fig); print("fig", name)


def load200():
    d = {}
    for a in COL:
        p = R / "final_200seed" / f"{'base_ovsplit' if a == 'ov_split' else a}.json"   # file keeps its historical name base_ovsplit
        if p.exists(): d[a] = json.loads(p.read_text())
    return d


def succ(d, a): return np.array([e["outcome"] == "success" for e in sorted(d[a]["episodes"], key=lambda e: e["seed"])])


def fig_headline(d):
    arms = [a for a in MAIN if a in d]
    if not arms: print("skip headline"); return
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); rows = []
    for i, a in enumerate(arms):
        k = int(succ(d, a).sum()); n = d[a]["n"]; lo, hi = wilson(k, n); p = k / n
        ax.bar(i, p, 0.55, color=COL[a]); ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="#2b2b29", lw=1.2, capsize=4)
        ax.text(i, hi + 0.02, f"{p:.3f}", ha="center", fontweight="bold")
        rows.append([NICE[a], n, f"{p:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
    if "stock_idm" in d:
        t = succ(d, "stock_idm").mean(); ax.axhline(t, color=COL["stock_idm"], ls="--", lw=1.4); ax.text(len(arms) - 0.5, t + 0.012, f"IDM teacher {t:.3f}", ha="right", color="#2b2b29", fontsize=8)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([NICE[a] for a in arms]); ax.set_ylim(0, 1.05); ax.set_ylabel("success rate (95% Wilson CI)")
    ax.set_title("Clear weather, density 0.05, 200 fresh seeds (5000-5199)", loc="left", fontsize=10); save(fig, "fig1_headline_success_n200")
    table("table1_headline_success_n200", ["policy", "n", "success", "95% CI"], rows, "Success on 200 fresh seeds, clear, density 0.05")


def fig_outcomes(d):
    arms = [a for a in MAIN if a in d]
    if not arms: return
    fig, ax = plt.subplots(figsize=(7.6, 3.6)); rows = []
    for i, a in enumerate(arms):
        oc = [e["outcome"] for e in d[a]["episodes"]]; n = len(oc); left = 0
        for k in ("success", "crash", "off_road", "timeout"):
            v = oc.count(k) / n
            if v > 0:
                ax.barh(i, v, left=left, color=OC[k], edgecolor="white", lw=1.5, label=k if i == 0 else None)
                if v >= 0.03: ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#2b2b29" if k == "success" else "white")
            left += v
        rows.append([NICE[a]] + [f"{oc.count(k)/n:.3f}" for k in OC])
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([NICE[a] for a in arms]); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of 200 episodes"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False); save(fig, "fig2_failure_modes_n200")
    table("table2_failure_modes_n200", ["policy", "success", "crash", "off_road", "timeout"], rows, "Episode outcomes, 200 seeds, clear, density 0.05")


def fig_efficiency(d):
    arms = [a for a in MAIN if a in d]
    if not arms: return
    mets = [("speed", "mean speed (m/s)"), ("wh_per_km", "energy (Wh/km)"), ("route", "route completion")]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6)); rows = []
    for ax, (m, lab) in zip(axs, mets):
        for i, a in enumerate(arms):
            x = np.array([e[m] for e in d[a]["episodes"]]); mu = x.mean(); se = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
            ax.errorbar(i, mu, yerr=se, fmt="o", color=COL[a], ms=9, capsize=4, lw=2); ax.text(i, mu + se + 0.04 * abs(mu) * 0.3, f"{mu:.2f}" if m != "wh_per_km" else f"{mu:.0f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(["Repo\nsparse", "Repo\nclear", "ov_split"][:len(arms)], fontsize=8); ax.set_title(lab, fontsize=9, loc="left"); ax.set_xlim(-0.6, len(arms) - 0.4)
    for a in arms:
        rows.append([NICE[a]] + [f"{np.mean([e[m] for e in d[a]['episodes']]):.3f}" for m, _ in mets])
    save(fig, "fig3_speed_energy_route_n200"); table("table3_speed_energy_route_n200", ["policy", "mean speed m/s", "Wh/km", "route completion"], rows, "Driving efficiency, 200 seeds, clear, density 0.05 (figure: dots = mean, whiskers = 95% CI)")


def fig_paired(d):
    if "ov_split" not in d: return
    rng = np.random.default_rng(0); rows = []; fig, ax = plt.subplots(figsize=(7.4, 3.2)); refs = [a for a in ("repo_bc_sparse", "repo_bc_clear", "stock_idm", "safe_teacher") if a in d]
    for i, b in enumerate(refs):
        x, y = succ(d, "ov_split"), succ(d, b); A, B = int((x & ~y).sum()), int((~x & y).sum()); p = binomtest(A, A + B, 0.5).pvalue if A + B else 1.0
        idx = rng.integers(0, len(x), (5000, len(x))); dd = x[idx].mean(1) - y[idx].mean(1); lo, hi = np.percentile(dd, [2.5, 97.5]); diff = x.mean() - y.mean()
        ax.errorbar(diff, i, xerr=[[diff - lo], [hi - diff]], fmt="o", color=COL[b], ms=7, capsize=4, lw=2); ax.text(hi + 0.012, i, f"{diff:+.3f}  (p={p:.3g})", va="center", fontsize=8)
        rows.append([NICE[b], f"{diff:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", A, B, f"{p:.4f}"])
    ax.axvline(0, color="#2b2b29", lw=1); ax.set_yticks(range(len(refs))); ax.set_yticklabels([f"vs {NICE[b]}" for b in refs]); ax.invert_yaxis()
    ax.set_xlabel("success(ov_split) - success(other), same 200 seeds"); ax.set_xlim(-0.12, 0.38); ax.grid(axis="y", visible=False); save(fig, "fig4_paired_difference_forest")
    table("table4_paired_ov_split_vs_others", ["ov_split vs", "diff", "95% paired-bootstrap CI", "ov_split-only wins", "other-only wins", "exact McNemar p"], rows, "Paired comparison on identical seeds (200)")


def fig_seed_strip(d):
    arms = [a for a in MAIN if a in d]
    if not arms: return
    fig, ax = plt.subplots(figsize=(10, 2.6)); codes = list(OC)
    for i, a in enumerate(arms):
        eps = sorted(d[a]["episodes"], key=lambda e: e["seed"])
        for j, e in enumerate(eps): ax.add_patch(plt.Rectangle((j, i), 1, 0.85, color=OC[e["outcome"]], lw=0))
    ax.set_xlim(0, len(eps)); ax.set_ylim(len(arms), 0); ax.set_yticks([i + 0.42 for i in range(len(arms))]); ax.set_yticklabels([NICE[a] for a in arms]); ax.set_xlabel("seed index 5000-5199"); ax.grid(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=OC[k]) for k in codes], labels=codes, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.3), frameon=False); save(fig, "fig5_per_seed_outcomes")


def fig_weather():
    arms = [a for a in MAIN if (R / f"sparse_traffic_weather_robustness_{a}_d0.05.json").exists()]
    if not arms: print("skip weather"); return
    tag = "_n200" if all((R / f"sparse_traffic_weather_robustness_{a}_d0.05_n200.json").exists() for a in arms) else ""
    data = {a: json.loads((R / f"sparse_traffic_weather_robustness_{a}_d0.05{tag}.json").read_text())["cells"] for a in arms}
    ws = list(next(iter(data.values())))
    lab = {"clear:0.0": "clear", "fog:1.0": "fog 1.0", "rain:0.6": "rain 0.6", "rain:1.0": "rain 1.0"}
    ret = json.loads((R / "retention_table.json").read_text()) if (R / "retention_table.json").exists() else {}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axs[0]
    for kind, ls in (("fog", "-"), ("rain", "--")):
        lv = sorted(float(k.split(":")[1]) for k in ret if k.startswith(kind)); vals = [min(ret[f"{kind}:{l}"]["retention"], 1.0) for l in lv]
        ax.plot([0] + lv, [1.0] + vals, ls, color="#2b2b29", marker="o", ms=4, lw=1.6); ax.text(lv[-1], vals[-1] + 0.05, kind, ha="right", fontsize=9)
    ax.set_xlabel("weather intensity"); ax.set_ylabel("share of vehicles still detected (capped at 1)"); ax.set_ylim(0, 1.1)
    ax.set_title("Stress applied: measured YOLO detection retention", loc="left", fontsize=10)
    ax = axs[1]; w = 0.8 / len(arms); rows = []
    for i, a in enumerate(arms):
        for j, k in enumerate(ws):
            c = data[a][k]; lo, hi = c["success_ci"]; x = j + i * w - 0.4 + w / 2
            ax.bar(x, c["success"], w - 0.03, color=COL[a], label=NICE[a] if j == 0 else None); ax.errorbar(x, c["success"], yerr=[[c["success"] - lo], [hi - c["success"]]], color="#2b2b29", lw=1, capsize=2)
            ax.text(x, hi + 0.012, f"{c['success']:.2f}", ha="center", fontsize=7)
            rows.append([NICE[a], lab.get(k, k), c["n"], f"{c['keep_prob']:.2f}", f"{c['success']:.3f}", f"[{lo:.2f}, {hi:.2f}]", f"{c['crash_or_offroad']:.3f}", f"{c['timeout']:.3f}", f"{c['retention_vs_clear']:.2f}"])
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f"{lab.get(k, k)}\nkeep {data[arms[0]][k]['keep_prob']:.2f}" for k in ws]); ax.set_ylim(0, 1.12); ax.set_ylabel("success rate (95% Wilson CI)")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07)); ax.set_title(f"Outcome under that stress (n={data[arms[0]][ws[0]]['n']} seeds/cell)", loc="left", fontsize=10)
    save(fig, "fig6_weather_stress_vs_success"); table("table5_weather_robustness", ["policy", "weather", "n", "detector keep prob", "success", "95% CI", "crash+off-road", "timeout", "retention vs clear"], rows, f"Weather robustness, density 0.05 (n per cell as listed{'' if tag else '; 40-seed block 1010-1049'})")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for a in arms:
        ys = [data[a][k]["retention_vs_clear"] for k in ws]; ax.plot(range(len(ws)), ys, marker="o", color=COL[a], lw=2); ax.text(len(ws) - 0.92, ys[-1], NICE[a], color="#2b2b29", fontsize=7, va="center")
    ax.axhline(1, color="#6b6b66", lw=0.8, ls=":"); ax.set_xticks(range(len(ws))); ax.set_xticklabels([lab.get(k, k) for k in ws]); ax.set_ylim(0.8, 1.15); ax.set_xlim(-0.2, len(ws) + 0.9)
    ax.set_ylabel("success(weather) / success(clear)"); save(fig, "fig7_weather_retention_ratio")


def fig_localisation():
    arms = [a for a in ("ov_split", "repo_bc_sparse", "base_a") if (R / f"localisation_error_sensitivity_{a}_d0.05.json").exists()]
    if not arms: print("skip localisation"); return
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); rows = []
    for a in arms:
        cells = sorted(json.loads((R / f"localisation_error_sensitivity_{a}_d0.05.json").read_text())["cells"].values(), key=lambda c: c["lat_m"])
        x = [c["lat_m"] for c in cells]; y = [c["success"] for c in cells]; lo = [c["success_ci"][0] for c in cells]; hi = [c["success_ci"][1] for c in cells]
        ax.plot(x, y, marker="o", color=COL[a], lw=2.4 if a == "ov_split" else 1.6); ax.fill_between(x, lo, hi, color=COL[a], alpha=0.12, lw=0); ax.text(x[-1] + 0.04, y[-1], NICE[a], fontsize=8, va="center")
        rows += [[NICE[a], c["lat_m"], c["heading_rad"], c["n"], f"{c['success']:.3f}", f"[{c['success_ci'][0]:.2f}, {c['success_ci'][1]:.2f}]"] for c in cells]
    ax.set_xlabel("injected HD-map localisation error: lateral (m) [heading error scaled alongside]"); ax.set_ylabel("success rate (n=40/cell, 95% CI band)"); ax.set_xlim(-0.05, 2.7); ax.set_ylim(0, 1.05)
    save(fig, "fig8_localisation_error_sensitivity"); table("table6_localisation_error", ["policy", "lateral err m", "heading err rad", "n", "success", "95% CI"], rows, "Sensitivity to map-localisation error, clear, density 0.05")


def fig_dagger():
    out = []
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for a in ("ov_split", "base_a"):
        p = Path(f"bev_rl/adaptive/runs/{a}/log.json")
        if not p.exists(): continue
        log = json.loads(p.read_text()); it = [r["iter"] for r in log]
        sr = [r["rollout_outcomes"].get("success", 0) / max(sum(r["rollout_outcomes"].values()), 1) for r in log]
        axs[0].plot(it, [r["train_loss"] for r in log], marker="o", color=COL[a], lw=2); axs[1].plot(it, sr, marker="o", color=COL[a], lw=2)
        axs[0].text(it[-1] + 0.1, log[-1]["train_loss"], NICE[a], fontsize=8, va="center"); axs[1].text(it[-1] + 0.1, sr[-1], NICE[a], fontsize=8, va="center")
        out += [[NICE[a], r["iter"], r["samples"], f"{r['train_loss']:.4f}", f"{s:.2f}"] for r, s in zip(log, sr)]
    axs[0].set_title("training loss per DAgger round", loc="left", fontsize=9); axs[1].set_title("rollout success per round (training mixture, includes teacher actions)", loc="left", fontsize=8)
    for ax in axs: ax.set_xlabel("DAgger round"); ax.set_xlim(-0.2, 6.3)
    save(fig, "fig9_dagger_training"); table("table7_dagger_rounds", ["policy", "round", "aggregate samples", "train loss", "rollout success"], out, "DAgger training rounds (rollout success mixes teacher and student actions, not a clean policy score)")


def main():
    d = load200()
    fig_headline(d); fig_outcomes(d); fig_efficiency(d); fig_paired(d); fig_seed_strip(d); fig_weather(); fig_localisation(); fig_dagger()


if __name__ == "__main__":
    main()
