"""Figures + tables for the sparse-traffic phase (2026-11-05 on). Reads measured files only; a missing input is skipped and printed, never faked.
Run from repo root:  venv/Scripts/python.exe paper/make_sparse_traffic_figures_and_tables.py
Outputs: paper/figures/fig8..fig10*.png, paper/tables/*.md and *.csv
Palette: dataviz reference categorical slots blue/orange/aqua/magenta (validated; contrast WARN -> every bar/line is directly labelled and a table exists)."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("bev_rl/adaptive/results"); L = Path("bev_rl/adaptive/logs"); F = Path("paper/figures"); T = Path("paper/tables")
F.mkdir(exist_ok=True, parents=True); T.mkdir(exist_ok=True, parents=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "axes.edgecolor": "#6b6b66", "text.color": "#2b2b29"})
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
NICE = {"repo_bc_sparse": "Repo BC (sparse traffic)", "repo_bc_clear": "Repo BC (clear, no traffic)", "base_a": "DAgger, GitHub BEV (base_a)", "ov_split": "DAgger, map-overlay BEV (ov_split)"}
WLAB = {"clear:0.0": "clear", "fog:1.0": "fog 1.0", "rain:0.6": "rain 0.6", "rain:1.0": "rain 1.0"}


def save_table(name, header, rows, caption):
    with open(T / f"{name}.csv", "w", newline="") as f: csv.writer(f).writerows([header] + rows)
    md = f"**{caption}**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"
    (T / f"{name}.md").write_text(md); print("table", name)


def fig_sparse_robustness():
    arms = [a for a in NICE if (R / f"sparse_traffic_weather_robustness_{a}_d0.05.json").exists()]
    if not arms: print("skip fig8 (no robustness results)"); return
    data = {a: json.load(open(R / f"sparse_traffic_weather_robustness_{a}_d0.05.json"))["cells"] for a in arms}
    ws = list(next(iter(data.values())).keys())
    fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.8 / len(arms); rows = []
    for i, a in enumerate(arms):
        for j, k in enumerate(ws):
            c = data[a][k]; lo, hi = c["success_ci"]
            ax.bar(j + i * w - 0.4 + w / 2, c["success"], w - 0.03, color=SER[i], label=NICE[a] if j == 0 else None)
            ax.errorbar(j + i * w - 0.4 + w / 2, c["success"], yerr=[[c["success"] - lo], [hi - c["success"]]], color="#2b2b29", lw=1, capsize=2)
            ax.text(j + i * w - 0.4 + w / 2, hi + 0.015, f"{c['success']:.2f}", ha="center", fontsize=7)
            rows.append([NICE[a], WLAB.get(k, k), c["n"], f"{c['keep_prob']:.2f}", f"{c['success']:.3f}", f"[{lo:.2f}, {hi:.2f}]", f"{c['crash_or_offroad']:.3f}", f"{c['timeout']:.3f}"])
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([WLAB.get(k, k) for k in ws]); ax.set_ylim(0, 1.12); ax.set_ylabel("success rate (95% Wilson CI)")
    spread = max(max(c["success"] for c in data[a].values()) - min(c["success"] for c in data[a].values()) for a in arms)
    n0 = next(iter(next(iter(data.values())).values()))["n"]
    ax.set_title(f"Sparse traffic (density 0.05), {n0} fresh seeds: largest success change across weathers for any policy = {spread:.3f}", fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32)); ax.grid(axis="y", color="#e8e8e4", lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(F / "fig8_sparse_weather_robustness.png"); plt.close(fig); print("fig8")
    save_table("table_sparse_weather_robustness", ["policy", "weather", "n", "detector keep prob", "success", "success 95% CI", "crash/off-road", "timeout"], rows,
               "Sparse traffic (density 0.05, seeds 1010-1049): success by policy and weather, with measured detector retention")


def fig_teacher_ceiling():
    rows, labels = [], {"stock": "stock IDM", "no_lane_change": "IDM, no lane change", "no_lc_defensive": "IDM, no lane change,\nlarger gap"}
    p = R / "expert_ceiling_lane_change_ablation_d0.2.json"; d = json.load(open(p)) if p.exists() else {}
    q = next(R.glob("safe_teacher_ceiling_d0.2*.json"), None)
    if q: d["safe_teacher"] = json.load(open(q))["safe_teacher"]; labels["safe_teacher"] = "IDM + predictive\nsafety layer"
    if not d: print("skip fig9"); return
    ks = list(d); fig, ax = plt.subplots(figsize=(8, 3.8)); bottom = np.zeros(len(ks))
    for key, col, nm in (("success", "#2a78d6", "success"), ("crash", "#eb6834", "crash"), ("timeout", "#8c8c86", "timeout")):
        v = np.array([d[k][key] for k in ks]); ax.bar(range(len(ks)), v, 0.6, bottom=bottom, color=col, edgecolor="#fcfcfb", lw=2, label=nm)
        for i, x in enumerate(v):
            if x > 0.06: ax.text(i, bottom[i] + x / 2, f"{x:.2f}", ha="center", va="center", color="white", fontsize=8)
        bottom += v
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([labels[k] for k in ks], fontsize=8); ax.set_ylabel("share of 20 episodes")
    best = max(ks, key=lambda k: d[k]["success"]); ax.set_title(f"Teacher ceiling in dense traffic (0.2): best variant = {labels[best].replace(chr(10), ' ')}, success {d[best]['success']:.2f}", fontsize=10, loc="left"); ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(F / "fig9_teacher_ceiling_dense_traffic.png"); plt.close(fig); print("fig9")
    save_table("table_teacher_ceiling_dense_traffic", ["teacher", "n", "success", "crash", "timeout"], [[labels[k].replace("\n", " "), d[k]["n"], d[k]["success"], d[k]["crash"], d[k]["timeout"]] for k in ks],
               "IDM teacher variants, clear weather, density 0.2, seeds 3100-3119")


def fig_ppo_sweep():
    runs = {}
    for n in ("rl_full", "rl_plain", "rl_energy", "rl_safe", "rl_head"):
        p = L / f"{n}.log"
        if p.exists():
            pts = [json.loads(l) for l in p.read_text().splitlines() if l.startswith('{"steps"')]
            if pts: runs[n] = pts
    if not runs: print("skip fig10"); return
    desc = {"rl_full": "energy + safe-speed + headway + crash-scaling", "rl_plain": "base reward", "rl_energy": "+ energy term", "rl_safe": "+ safe-speed, crash-scaling", "rl_head": "+ headway term"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True); cols = dict(zip(runs, SER + ["#8c8c86"])); rows = []
    for n, pts in runs.items():
        s = [p["steps"] / 1000 for p in pts]
        axs[0].plot(s, [p["val_success"] for p in pts], "o-", color=cols[n], lw=1.8, ms=4, label=n); axs[1].plot(s, [p["wh_per_km"] for p in pts], "o-", color=cols[n], lw=1.8, ms=4)
        rows.append([n, desc[n], f"{pts[0]['val_success']:.3f}", f"{pts[-1]['val_success']:.3f}", f"{max(p['val_success'] for p in pts):.3f}", f"{pts[-1]['wh_per_km']:.0f}"])
    axs[0].set_ylabel("validation success (n = 40)"); axs[0].set_ylim(0, 0.6); axs[1].set_ylabel("Wh/km"); axs[0].legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    for a in axs: a.set_xlabel("PPO steps (thousands)"); a.grid(color="#e8e8e4", lw=0.6)
    gain = max(max(p["val_success"] for p in pts) - pts[0]["val_success"] for pts in runs.values()); fin = max(pts[-1]["val_success"] - pts[0]["val_success"] for pts in runs.values())
    fig.suptitle(f"PPO fine-tuning of ov_split (dense traffic): start {next(iter(runs.values()))[0]['val_success']:.3f}; best gain any run {gain:+.3f}, best final gain {fin:+.3f}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig10_ppo_reward_sweep.png"); plt.close(fig); print("fig10")
    save_table("table_ppo_reward_sweep", ["run", "reward terms", "val success @0", "val success @150k", "best", "Wh/km @150k"], rows, "PPO reward-combination sweep (single seed, validation seeds 3100-3109, density 0.2)")



def fig_localisation_sensitivity():
    arms = [a for a in ("ov_split", "base_a", "repo_bc_sparse") if (R / f"localisation_error_sensitivity_{a}_d0.05.json").exists()]
    if not arms: print("skip fig11"); return
    data = {a: json.load(open(R / f"localisation_error_sensitivity_{a}_d0.05.json"))["cells"] for a in arms}
    keys = list(next(iter(data.values())).keys()); xl = [f"{data[arms[0]][k]['lat_m']:g} m / {data[arms[0]][k]['heading_rad']:g} rad" for k in keys]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9)); rows = []
    for a in arms:
        col = SER[list(NICE).index(a)]; c = [data[a][k] for k in keys]; x = np.arange(len(keys))
        axs[0].errorbar(x, [v["success"] for v in c], yerr=[[v["success"] - v["success_ci"][0] for v in c], [v["success_ci"][1] - v["success"] for v in c]], fmt="o-", color=col, lw=1.8, ms=4, capsize=2, label=NICE[a])
        axs[1].plot(x, [v["off_road"] for v in c], "o-", color=col, lw=1.8, ms=4, label=NICE[a])
        for k, v in zip(keys, c): rows.append([NICE[a], f"{v['lat_m']:g}", f"{v['heading_rad']:g}", v["n"], f"{v['success']:.3f}", f"[{v['success_ci'][0]:.2f}, {v['success_ci'][1]:.2f}]", f"{v['crash']:.3f}", f"{v['off_road']:.3f}", f"{v['timeout']:.3f}"])
    for ax, yl in zip(axs, ("success rate (95% Wilson CI)", "off-road rate")):
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(xl, fontsize=7, rotation=20); ax.set_ylabel(yl); ax.set_ylim(0, 1.02); ax.grid(color="#e8e8e4", lw=0.6)
        ax.set_xlabel("persistent pose error (lateral, heading)")
    axs[0].legend(fontsize=7, frameon=False, loc="lower left")
    drop = {a: data[a][keys[0]]["success"] - data[a][keys[-1]]["success"] for a in arms}
    fig.suptitle("Sparse traffic, clear weather, n=40: success loss from zero to largest pose error: " + ", ".join(f"{a} {drop[a]:.2f}" for a in arms), fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(F / "fig11_localisation_error_sensitivity.png"); plt.close(fig); print("fig11")
    save_table("table_localisation_error_sensitivity", ["policy", "lateral error (m)", "heading error (rad)", "n", "success", "success 95% CI", "crash", "off-road", "timeout"], rows,
               "Sparse traffic (0.05), clear weather, seeds 1010-1049: persistent ego-pose error against the HD map (overlay policy: map drawn from erroneous pose; scalar policies: error added to lateral/heading inputs)")


if __name__ == "__main__":
    fig_sparse_robustness(); fig_teacher_ceiling(); fig_ppo_sweep(); fig_localisation_sensitivity()
