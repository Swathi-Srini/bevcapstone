"""Paired analysis of results/final_200seed/*.json (identical seeds): per-arm Wilson CI + means, pairwise exact McNemar (success) and paired bootstrap of the success difference.
Writes paper/tables/table_final_200seed.{csv,md}.   python -m bev_rl.adaptive.paired_stats_final_200seed"""
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from bev_rl.adaptive.eval_caution import wilson
D = Path("bev_rl/adaptive/results/final_200seed"); ORDER = ["safe_teacher", "base_ovsplit", "ctrl_nocaut", "stock_idm", "repo_bc_clear", "repo_bc_sparse"]
R = {a: json.loads((D / f"{a}.json").read_text()) for a in ORDER}
S = {a: np.array([e["outcome"] == "success" for e in sorted(R[a]["episodes"], key=lambda e: e["seed"])]) for a in ORDER}
md = ["| arm | n | success [95% CI] | crash | off-road | timeout | speed m/s | Wh/km | route |", "|---|---|---|---|---|---|---|---|---|"]
for a in ORDER:
    r = R[a]; k = int(S[a].sum()); lo, hi = wilson(k, r["n"])
    md.append(f"| {a} | {r['n']} | {r['success']:.3f} [{lo:.3f}, {hi:.3f}] | {r['crash']:.3f} | {r['off_road']:.3f} | {r['timeout']:.3f} | {r['mean_speed']:.2f} | {r['wh_per_km']:.1f} | {r['route']:.3f} |")
md += ["", "Paired (same seeds): diff = success(A) - success(B); b/c = discordant seeds (A only / B only); exact McNemar p; 95% paired-bootstrap CI.", "",
       "| A | B | diff | A-only | B-only | p | boot CI |", "|---|---|---|---|---|---|---|"]
rng = np.random.default_rng(0)
for A, B in itertools.combinations(ORDER, 2):
    x, y = S[A], S[B]; b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    idx = rng.integers(0, len(x), (5000, len(x))); d = (x[idx].mean(1) - y[idx].mean(1))
    md.append(f"| {A} | {B} | {x.mean()-y.mean():+.3f} | {b} | {c} | {p:.4f} | [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] |")
Path("paper/tables").mkdir(parents=True, exist_ok=True); Path("paper/tables/table_final_200seed.md").write_text("\n".join(md)); print("\n".join(md))
