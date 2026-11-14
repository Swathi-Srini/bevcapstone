"""Aggregate camera-pipeline evaluation episodes (read-only): outcomes with Wilson 95% intervals, completion time, speed, estimated energy, and perception metrics against simulator truth.

    python -m bev_rl.adaptive.summarise_camera_eval bev_rl/adaptive/results/camera_eval_originals
Writes <dir>/summary.md and <dir>/episodes.csv. Perception metrics (vehicles inside the BEV window only): recall = vehicles with a corrected object within 3 m / vehicles; in-view recall counts only
vehicles inside some camera's field of view; position error of matched objects; false objects per frame; share of vehicle-containing frames whose BEV equals the ground-truth BEV exactly.
"""
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)


def main():
    root = Path(sys.argv[1]); E = [json.loads(f.read_text()) for f in sorted((root / "episodes").glob("*.json"))]
    G = defaultdict(list)
    for e in E: G[(e["arm"], e["weather"])].append(e)
    lines = ["| model | weather | n | success [95% CI] | collision | off-road | timeout | completion s | speed m/s | Wh/km | route | vehicles seen/true (recall) | in-view recall | median pos. error m | false obj/frame | exact BEV on vehicle frames |", "|" + "---|" * 16]
    with open(root / "episodes.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "weather", "seed", "outcome", "steps", "completion_s", "speed_mps", "wh_per_km", "route", "c_input", "wall_s", "video"])
        for e in E: w.writerow([e["arm"], e["weather"], e["seed"], e["outcome"], e["steps"], e["steps"] / 10 if e["outcome"] == "success" else "", round(e["speed"], 3), round(e["wh_per_km"], 2), round(e["route"], 3), e.get("c_input"), e.get("wall_s"), e.get("video")])
    for (arm, wx), R in sorted(G.items()):
        n = len(R); oc = [r["outcome"] for r in R]; ok = [r for r in R if r["outcome"] == "success"]; lo, hi = wilson(len(ok), n)
        S = [s for r in R for s in r["steps_log"]]; truth = sum(s["n_truth"] for s in S); inview = sum(s["n_truth_in_view"] for s in S); m = sum(s["corr_matched"] for s in S); mv = sum(s["corr_matched_in_view"] for s in S)
        err = [math.hypot(dx, dy) for s in S for dx, dy in zip(s["corr_dx"], s["corr_dy"])]; fp = sum(s["corr_fp"] + s["corr_dup"] for s in S) / max(1, len(S))
        vf = [s for s in S if s["n_truth"] > 0]; eq = sum(s["corr_grid_equal"] for s in vf)
        lines.append(f"| {arm} | {wx} | {n} | {len(ok)}/{n} [{lo:.2f}, {hi:.2f}] | {oc.count('crash')} | {oc.count('off_road')} | {oc.count('timeout')} | {np.mean([r['steps'] / 10 for r in ok]) if ok else float('nan'):.1f} | "
                     f"{np.mean([r['speed'] for r in R]):.2f} | {np.mean([r['wh_per_km'] for r in R]):.1f} | {np.mean([r['route'] for r in R]):.2f} | {m}/{truth} ({m / max(1, truth):.3f}) | {mv}/{inview} ({mv / max(1, inview):.3f}) | "
                     f"{np.median(err) if err else float('nan'):.2f} | {fp:.3f} | {eq}/{len(vf)} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
