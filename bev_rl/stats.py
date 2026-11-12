"""Paired statistics for comparing policies on identical scenarios."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_continuous(a, b, n_boot: int = 10_000, seed: int = 0) -> dict:
    """b - a over scenarios where both are finite: mean diff, bootstrap 95% CI, Wilcoxon p, Cohen's dz."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = (b - a)[keep]
    n = int(d.size)
    if n < 2:
        return {"n": n, "mean_a": float("nan"), "mean_b": float("nan"), "mean_diff": float("nan"),
                "ci95": (float("nan"), float("nan")), "wilcoxon_p": float("nan"), "dz": float("nan")}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    nonzero = np.any(d != 0)
    p = float(stats.wilcoxon(d).pvalue) if nonzero else 1.0
    sd = d.std(ddof=1)
    return {"n": n, "mean_a": float(a[keep].mean()), "mean_b": float(b[keep].mean()), "mean_diff": float(d.mean()),
            "ci95": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
            "wilcoxon_p": p, "dz": float(d.mean() / sd) if sd > 0 else float("nan")}


def paired_binary(a, b) -> dict:
    """Exact McNemar on discordant pairs (a, b are 0/1 per scenario) plus Clopper-Pearson CIs."""
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    only_a, only_b = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    disc = only_a + only_b
    p = float(stats.binomtest(only_a, disc, 0.5).pvalue) if disc > 0 else 1.0
    def cp(x):
        r = stats.binomtest(int(x.sum()), len(x)).proportion_ci(0.95, method="exact")
        return (float(r.low), float(r.high))
    return {"n": int(a.size), "rate_a": float(a.mean()), "rate_b": float(b.mean()), "only_a": only_a,
            "only_b": only_b, "mcnemar_exact_p": p, "ci95_a": cp(a), "ci95_b": cp(b)}
