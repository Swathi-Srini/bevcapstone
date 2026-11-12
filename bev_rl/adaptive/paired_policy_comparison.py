"""Paired comparison of policies evaluated on IDENTICAL seeds (exact McNemar test + paired bootstrap CI of the success difference).
Reads per-episode outcomes from results/*.json (no simulation). Same seeds => comparing two policies is far more powerful than comparing two overlapping Wilson CIs.
    python -m bev_rl.adaptive.paired_policy_comparison            # prints and writes paper/tables/table_paired_policy_comparison.{md,csv}
Conventions: A vs B; b = seeds A succeeds & B fails; c = seeds A fails & B succeeds; p = exact two-sided binomial (McNemar) on (b, c); diff = success(A) - success(B).
"""
import csv, json
from math import comb
from pathlib import Path
import numpy as np

R = Path("bev_rl/adaptive/results"); T = Path("paper/tables"); T.mkdir(parents=True, exist_ok=True)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_boot(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(x, float) - np.asarray(y, float); m = len(d)
    bs = [d[rng.integers(0, m, m)].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_cells():
    """-> {(condition, arm): {seed: success_bool}} from the weather-robustness and localisation-error result files."""
    cells = {}
    for f in R.glob("sparse_traffic_weather_robustness_*_d0.05.json"):
        j = json.load(open(f))
        for e in j["episodes"]: cells.setdefault((f"weather {e['weather']}", j["arm"]), {})[e["seed"]] = e["outcome"] == "success"
    for f in R.glob("localisation_error_sensitivity_*_d0.05.json"):
        j = json.load(open(f))
        for e in j["episodes"]: cells.setdefault((f"pose error {e['lat_m']:g} m / {e['heading_rad']:g} rad", j["arm"]), {})[e["seed"]] = e["outcome"] == "success"
    for f in (R / "caution").glob("*.json"):          # caution-study arms evaluated at sparse density (same seeds/protocol)
        j = json.load(open(f))
        if j["args"].get("density") != 0.05 or j["args"].get("n") != 40 or "sweep" in f.name: continue
        for e in j["episodes"]: cells.setdefault((f"weather {e['weather']}", j["arm"]), {})[e["seed"]] = e["outcome"] == "success"
    return cells


def main():
    cells = load_cells(); conds = sorted({c for c, _ in cells}, key=lambda s: (s.split()[0], float(s.split()[2]) if s.startswith("pose") else 0, s))
    arms = sorted({a for _, a in cells}); rows = []
    pairs = [(a, b) for a in arms for b in arms if a < b or False]
    for cond in conds:
        have = [a for a in arms if (cond, a) in cells]
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                sa, sb = cells[(cond, a)], cells[(cond, b)]; seeds = sorted(set(sa) & set(sb))
                if len(seeds) < 10: continue
                x = [sa[s] for s in seeds]; y = [sb[s] for s in seeds]
                bb = sum(1 for u, v in zip(x, y) if u and not v); cc = sum(1 for u, v in zip(x, y) if v and not u)
                d, lo, hi = paired_boot(x, y); rows.append([cond, a, b, len(seeds), f"{np.mean(x):.3f}", f"{np.mean(y):.3f}", f"{d:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", bb, cc, f"{mcnemar_exact(bb, cc):.4f}"])
    header = ["condition", "policy A", "policy B", "n seeds", "success A", "success B", "A - B", "paired 95% CI", "A only", "B only", "McNemar p"]
    with open(T / "table_paired_policy_comparison.csv", "w", newline="") as f: csv.writer(f).writerows([header] + rows)
    md = "**Paired policy comparison on identical seeds (sparse traffic 0.05, seeds 1010-1049)**\n\n| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"
    (T / "table_paired_policy_comparison.md").write_text(md)
    print(md)

if __name__ == "__main__":
    main()
