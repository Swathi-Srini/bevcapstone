"""PAPER ANALYSIS of the paired A (non-adaptive) vs C (adaptive) camera evaluation.   NEW FILE (2026-11-16).  Read-only on the episode JSONs.

    python -m bev_rl.adaptive.analyze_severity_eval --episodes DIR [DIR ...] --out OUTDIR [--label "20 paired seeds (pilot)"]

Only complete pairs (A and C both present for the same condition and seed) enter the paired statistics.  Outputs in OUTDIR: summary.json, tables.md, table_main.tex, fig_speed_energy.png.
Statistics: exact two-sided McNemar on discordant pairs, Wilson 95% intervals, cluster (seed) bootstrap of the paired mean (2,000 resamples, seed 20261116).
Energy: paired Wh/km on pairs where BOTH arms succeeded (primary); all-episode Wh/km reported separately.  Pilot scale: no formal non-inferiority claim.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

CONDS = ["clear:0.0", "fog:1.0", "rain:1.0"]
NAMES = {"clear:0.0": "Clear", "fog:1.0": "Fog 1.0", "rain:1.0": "Rain 1.0"}


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(x, rng, B=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (float("nan"), float("nan"))
    m = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def brake_share(tr):
    v = np.asarray(tr["speed"], float)
    return float(np.mean(np.diff(v) / 0.1 < -2.0)) if len(v) > 1 else float("nan")


def load(dirs):
    R = {}
    for d in dirs:
        for f in Path(d).glob("episodes/*.json"):
            e = json.loads(f.read_text()); R[(e["arm"], e["weather"], e["seed"])] = e
    return R


def trunc(e, H):
    """Outcome at horizon H derived from the trace (an episode still running at step H counts as timeout)."""
    if e["steps"] <= H: return e["outcome"]
    return "timeout"


def mean(x): return float(np.mean(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--label", default="")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); R = load(a.episodes); rng = np.random.default_rng(20261116)
    S = {"label": a.label, "conditions": {}}; md = [f"# Paired A vs C evaluation {a.label}\n"]
    for c in CONDS:
        seeds = sorted(s for (arm, w, s) in R if arm == "A" and w == c and ("C", c, s) in R)
        A = [R[("A", c, s)] for s in seeds]; C = [R[("C", c, s)] for s in seeds]; n = len(seeds)
        if n == 0: continue
        sa = np.array([e["outcome"] == "success" for e in A]); sc = np.array([e["outcome"] == "success" for e in C])
        b = int((sa & ~sc).sum()); cc = int((~sa & sc).sum()); both = sa & sc
        unsafe = lambda L: sum(e["outcome"] in ("crash", "off_road") for e in L)
        wa = np.array([e["wh_per_km"] for e in A]); wc = np.array([e["wh_per_km"] for e in C]); d = (wc - wa)[both]
        row = {"n_pairs": n, "seeds": [seeds[0], seeds[-1]],
               "success_A": int(sa.sum()), "success_C": int(sc.sum()), "success_A_ci": wilson(int(sa.sum()), n), "success_C_ci": wilson(int(sc.sum()), n), "mcnemar_A_only": b, "mcnemar_C_only": cc, "mcnemar_p": mcnemar_exact(b, cc),
               "outcomes_A": {k: sum(e["outcome"] == k for e in A) for k in ("success", "crash", "off_road", "timeout")}, "outcomes_C": {k: sum(e["outcome"] == k for e in C) for k in ("success", "crash", "off_road", "timeout")},
               "unsafe_A": unsafe(A), "unsafe_C": unsafe(C),
               "route_completion_A": mean([e["route_completion"] for e in A]), "route_completion_C": mean([e["route_completion"] for e in C]),
               "mean_speed_A": mean([e["mean_speed"] for e in A]), "mean_speed_C": mean([e["mean_speed"] for e in C]),
               "duration_s_A": mean([e["duration_s"] for e in A]), "duration_s_C": mean([e["duration_s"] for e in C]),
               "min_gap_A": mean([e["min_gap_m"] for e in A if e["min_gap_m"] is not None]), "min_gap_C": mean([e["min_gap_m"] for e in C if e["min_gap_m"] is not None]),
               "brake_share_A": mean([brake_share(e["trace"]) for e in A]), "brake_share_C": mean([brake_share(e["trace"]) for e in C]),
               "n_both_success": int(both.sum()), "whkm_A_both": mean(wa[both]), "whkm_C_both": mean(wc[both]), "whkm_diff_both": mean(d), "whkm_diff_both_ci": boot_ci(d, rng),
               "whkm_pct_change_both": float(100 * mean(d) / mean(wa[both])) if both.any() else float("nan"),
               "whkm_A_all": mean(wa), "whkm_C_all": mean(wc), "energy_wh_A_all": mean([e["energy_wh"] for e in A]), "energy_wh_C_all": mean([e["energy_wh"] for e in C]),
               "c_hat_mean_C": mean([np.mean(e["trace"]["c_hat"]) for e in C]), "c_true": C[0]["c_true"],
               "clear_false_caution_rate_C": mean([np.mean(np.asarray(e["trace"]["c_hat"]) > 0.05) for e in C]) if c == "clear:0.0" else None,
               "horizon800": {"success_A": sum(trunc(e, 800) == "success" for e in A), "success_C": sum(trunc(e, 800) == "success" for e in C)}}
        S["conditions"][c] = row
        md.append(f"## {NAMES[c]} (n = {n} paired seeds {seeds[0]}-{seeds[-1]})\n")
        md.append(f"| Metric | A (non-adaptive) | C (adaptive) |\n|---|---|---|")
        md.append(f"| Success | {row['success_A']}/{n} ({row['success_A_ci'][0]:.2f}-{row['success_A_ci'][1]:.2f}) | {row['success_C']}/{n} ({row['success_C_ci'][0]:.2f}-{row['success_C_ci'][1]:.2f}) |")
        md.append(f"| Outcomes (succ/crash/off-road/timeout) | {'/'.join(str(row['outcomes_A'][k]) for k in ('success','crash','off_road','timeout'))} | {'/'.join(str(row['outcomes_C'][k]) for k in ('success','crash','off_road','timeout'))} |")
        md.append(f"| Collisions + off-road | {row['unsafe_A']} | {row['unsafe_C']} |")
        md.append(f"| Route completion | {row['route_completion_A']:.3f} | {row['route_completion_C']:.3f} |")
        md.append(f"| Mean speed (m/s) | {row['mean_speed_A']:.2f} | {row['mean_speed_C']:.2f} |")
        md.append(f"| Duration (s) | {row['duration_s_A']:.1f} | {row['duration_s_C']:.1f} |")
        md.append(f"| Min bumper gap (m, privileged log) | {row['min_gap_A']:.1f} | {row['min_gap_C']:.1f} |")
        md.append(f"| Hard-braking share of steps | {row['brake_share_A']:.3f} | {row['brake_share_C']:.3f} |")
        md.append(f"| Wh/km, both-success pairs (n={row['n_both_success']}) | {row['whkm_A_both']:.1f} | {row['whkm_C_both']:.1f} |")
        md.append(f"| Wh/km, all episodes | {row['whkm_A_all']:.1f} | {row['whkm_C_all']:.1f} |")
        md.append(f"| Estimated caution (mean, true = {row['c_true']:.2f}) | n/a | {row['c_hat_mean_C']:.2f} |")
        md.append(f"\nMcNemar (A-only success {b}, C-only success {cc}): exact p = {row['mcnemar_p']:.3f}. Paired Wh/km difference C-A on both-success pairs: {row['whkm_diff_both']:.1f} "
                  f"(95% bootstrap CI {row['whkm_diff_both_ci'][0]:.1f} to {row['whkm_diff_both_ci'][1]:.1f}), {row['whkm_pct_change_both']:.1f}%.\n")
    (a.out / "summary.json").write_text(json.dumps(S, indent=1)); (a.out / "tables.md").write_text("\n".join(md))
    tex = ["\\begin{tabular}{lcccccc}", "\\hline", "Condition & Model & Success & Unsafe & Speed (m/s) & Wh/km (both) & $\\Delta$ Wh/km (\\%) \\\\", "\\hline"]
    for c, r in S["conditions"].items():
        n = r["n_pairs"]
        tex.append(f"{NAMES[c]} & A & {r['success_A']}/{n} & {r['unsafe_A']} & {r['mean_speed_A']:.2f} & {r['whkm_A_both']:.1f} & -- \\\\")
        tex.append(f" & C & {r['success_C']}/{n} & {r['unsafe_C']} & {r['mean_speed_C']:.2f} & {r['whkm_C_both']:.1f} & {r['whkm_pct_change_both']:+.1f} \\\\")
    tex += ["\\hline", "\\end{tabular}"]; (a.out / "table_main.tex").write_text("\n".join(tex))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.4)); xs = np.arange(len(S["conditions"])); lab = [NAMES[c] for c in S["conditions"]]
        for i, (key, ttl) in enumerate([("mean_speed", "Mean speed (m/s)"), ("whkm", "Wh/km (both-success pairs)"), ("success", "Success rate")]):
            va = [(r["success_A"] / r["n_pairs"] if key == "success" else r["whkm_A_both"] if key == "whkm" else r["mean_speed_A"]) for r in S["conditions"].values()]
            vc = [(r["success_C"] / r["n_pairs"] if key == "success" else r["whkm_C_both"] if key == "whkm" else r["mean_speed_C"]) for r in S["conditions"].values()]
            ax[i].bar(xs - 0.2, va, 0.38, label="A non-adaptive", color="#8a8f98"); ax[i].bar(xs + 0.2, vc, 0.38, label="C adaptive", color="#1b6ec2")
            ax[i].set_xticks(xs); ax[i].set_xticklabels(lab); ax[i].set_title(ttl, fontsize=10); ax[i].spines[["top", "right"]].set_visible(False)
        ax[0].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(a.out / "fig_speed_energy.png", dpi=200)
    except Exception as ex:
        print("figure skipped:", ex)
    print("\n".join(md))


if __name__ == "__main__":
    main()
