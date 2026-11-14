"""Re-derive the camera grid builder's geometric corrections from the stored controlled-placement files (offline, numpy only; no simulator, no camera).

Why this file exists: the original fitting script was not saved, so the constants in bev_state/camera_grid_builder.py (FRONT_AFFINE, SIDE_AFFINE) could not be reproduced from the repository.
This script reads bev_rl/adaptive/results/camera_geometry/placements_seed{2000,2001,2002}.jsonl (written by calibrate_camera_geometry.py: one vehicle placed at a KNOWN position, the unmodified
perception chain's reported objects stored next to the true centre) and re-fits the corrections by least squares, then reports how close the result is to the constants in the builder.

Pairing rule (documented here because it was not recorded before): for every placement and camera group, take the reported object nearest to the true centre -- for the side group AFTER the point
reflection (x, y) -> (-x, -y) -- and keep the pair only if that distance is below PAIR_MAX_M (the 'consistency within 6 m' filter) and the true centre lies inside the BEV window.
Front group = front_left_camera (stereo objects). Side group = left_camera and right_camera. The rear camera had too few pairs and is not fitted.

    python -m bev_rl.adaptive.fit_camera_corrections

First run (14 Nov 2026): 72 pairs (front 49, side 23) against the recorded 85 (52 / 33). Front coefficients agree with the builder's constants to 0.084 (error 1.61 -> 0.95 m, recorded 1.58 -> 0.91);
side coefficients agree only to 0.227 (error 2.66 -> 1.04 m, recorded 2.71 -> 1.30). Sixteen pairing variants and adding the 3100-3102 placement files did not reproduce 52 / 33, so the ORIGINAL fit cannot be
recovered exactly. The constants in bev_state/camera_grid_builder.py are unchanged and frozen. See docs/GEOMETRIC_CORRECTIONS.md section 5.
"""
import json
from pathlib import Path

import numpy as np

PAIR_MAX_M = 6.0
WINDOW = (-10.0, 10.0, -2.5, 17.5)               # x_min, x_max, y_min, y_max (BEV window)
FILES = [f"bev_rl/adaptive/results/camera_geometry/placements_seed{s}.jsonl" for s in (2000, 2001, 2002)]
BUILDER_FRONT = np.array([[1.15, 0.02, -0.07], [0.11, 1.03, 0.87]])
BUILDER_SIDE = np.array([[1.20, -0.04, -1.50], [0.02, 1.67, 0.27]])


def load():
    rows = []
    for f in FILES:
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    return rows


def pairs(rows):
    front, side = [], []
    for r in rows:
        tx, ty = r["true_x"], r["true_y"]
        if not (WINDOW[0] <= tx <= WINDOW[1] and WINDOW[2] <= ty <= WINDOW[3]):
            continue
        for grp, cams, flip, out in (("front", ("front_left_camera",), 1.0, front), ("side", ("left_camera", "right_camera"), -1.0, side)):
            cand = [(np.hypot(flip * o["x"] - tx, flip * o["y"] - ty), o) for o in r["objects"] if o["camera"] in cams]
            if not cand:
                continue
            d, o = min(cand, key=lambda c: c[0])
            if d < PAIR_MAX_M:
                out.append((flip * o["x"], flip * o["y"], tx, ty, r["seed"]))
    return np.array(front), np.array(side)


def fit(P):
    """Least squares [x, y, 1] -> [x_true, y_true]; returns the 2x3 matrix."""
    X = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    return np.linalg.lstsq(X, P[:, 2:4], rcond=None)[0].T


def loso(P):
    """Leave-one-scene-out median Euclidean error of the fitted correction."""
    err = []
    for s in np.unique(P[:, 4]):
        tr, te = P[P[:, 4] != s], P[P[:, 4] == s]
        A = fit(tr); pred = np.c_[te[:, 0], te[:, 1], np.ones(len(te))] @ A.T
        err += list(np.hypot(*(pred - te[:, 2:4]).T))
    return float(np.median(err))


def main():
    rows = load(); front, side = pairs(rows)
    print(f"placements: {len(rows)} | usable pairs: front {len(front)}, side {len(side)} (earlier record: 85 total = front 52 + side 33)")
    for name, P, ref in (("FRONT", front, BUILDER_FRONT), ("SIDE (after reflection)", side, BUILDER_SIDE)):
        A = fit(P); raw = np.median(np.hypot(P[:, 0] - P[:, 2], P[:, 1] - P[:, 3]))
        print(f"\n{name}: n={len(P)}")
        print("  re-fitted:\n", np.round(A, 2)); print("  builder constants:\n", ref)
        print(f"  max |difference| in coefficients: {np.abs(A - ref).max():.3f}")
        print(f"  median error before correction {raw:.2f} m; leave-one-scene-out after correction {loso(P):.2f} m")


if __name__ == "__main__":
    main()
