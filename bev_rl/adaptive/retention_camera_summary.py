"""RETENTION TABLE + COVERAGE SUMMARY for the camera retention pass.  NEW FILE; reads the episode files written by measure_retention_camera.py, writes ONLY into <out>/summary/ and never overwrites.

Definitions (approved 2026-11-15; the retired box-count retention_table.json is NOT used):
    matched(cond)        vehicles of the identical frame set matched (greedy 1-to-1, 3.0 m) by a corrected camera object inside the window
    retention(cond)      pooled  sum matched(cond) / sum matched(clear)           -> CALIBRATION TARGET        c = clip(1 - retention, 0, 1)
    paired-identity ret. |M_clear ∩ M_cond| / sum matched(clear)  (the SAME vehicle must be matched in clear and in cond)   -> separate metric, never a calibration target
    recall               sum matched(cond) / sum truth vehicles in the window      -> separate metric (clear row = the detection ceiling)
    duplicates, false positives, invalid-depth drops: counted separately (per frame and per clear-matched vehicle); never merged into recall or retention
    paired measurements  per frame (matched_cond - matched_clear) and per episode (episode retention), alongside the pooled ratio
Uncertainty: 2,000 episode-clustered bootstrap resamples (fixed seed 20261115), 95% percentile intervals; ONE shared index matrix for all conditions and metrics.
    A resample whose clear-matched total is zero has no ratio: it is EXCLUDED and counted.  Valid resamples < 1,900 -> bootstrap_unreliable.  c-CI half-width > 0.15 -> assessable false.
Coverage (approved): >= 200 clear matched vehicles AND >= 12 contributing episodes, evaluated on the first 20 training episodes (reserve stop rule) and on all available episodes.
Probe conditions (fog 0.45, rain 0.3) are measured and reported but never enter table_c or any fit.

    python -m bev_rl.adaptive.retention_camera_summary summarize --out bev_rl/adaptive/results/retention_camera [--name retention_table_camera.json] [--include-reserve]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from bev_rl.adaptive.measure_retention_camera import COND_NAMES, CONDITIONS, PROBES, cname, partition

BOOT_N, BOOT_SEED, MIN_VALID_RESAMPLES, CI_HALF_WIDTH_LIMIT = 2000, 20261115, 1900, 0.15
MIN_CLEAR_MATCHED, MIN_EPISODES = 200, 12


def training_view(out, include_reserve=False):
    """The ONLY loader a table / fit / Gate-2 script may use: training (and, if explicitly asked, reserve) episodes. Held-out episodes are never returned; they carry no retention records by design."""
    eps = []
    for f in sorted((Path(out) / "episodes").glob("s*.json")):
        e = json.loads(f.read_text())
        if e.get("status") != "complete": continue
        if e["partition"] == "train" or (include_reserve and e["partition"] == "reserve"): eps.append(e)
        elif e["partition"] not in ("heldout", "reserve"): raise SystemExit(f"{f}: unknown partition {e['partition']}")
    return eps


def _pct(x):
    lo, hi = np.percentile(x, [2.5, 97.5]); return [float(lo), float(hi)]


def retention_cells(episodes, boot=True):
    """Per-condition cells from the retention frames of `episodes` (loaded training episode dicts).  Gate 2 imports this to rebuild a raw cell from any subset of episodes.  -> {cond: {...}}"""
    E = len(episodes); z = lambda: np.zeros(E)
    C, TR, FRM = z(), z(), z()
    M = {n: z() for n in COND_NAMES}; P = {n: z() for n in COND_NAMES}; D = {n: z() for n in COND_NAMES}; F = {n: z() for n in COND_NAMES}; X = {n: z() for n in COND_NAMES}
    frame_delta = {n: [] for n in COND_NAMES}; frame_ratio = {n: [] for n in COND_NAMES}
    for i, e in enumerate(episodes):
        for fr in e["retention"]:
            FRM[i] += 1; TR[i] += fr["n_truth"]; c0 = fr["cond"]["clear:0.0"]; C[i] += c0["matched"]; base = set(c0["matched_idx"])
            for n in COND_NAMES:
                cell = fr["cond"][n]; M[n][i] += cell["matched"]; P[n][i] += len(base & set(cell["matched_idx"])); D[n][i] += cell["dup"]; F[n][i] += cell["fp"]; X[n][i] += cell["n_dropped_invalid_depth"]
                frame_delta[n].append(cell["matched"] - c0["matched"])
                if c0["matched"] > 0: frame_ratio[n].append(cell["matched"] / c0["matched"])
    idx = np.random.default_rng(BOOT_SEED).integers(0, E, (BOOT_N, E)) if (boot and E) else None            # ONE index matrix shared by every condition and metric
    out = {}
    for kind, level in CONDITIONS:
        n = cname(kind, level); c_tot, m_tot, p_tot = C.sum(), M[n].sum(), P[n].sum(); tr_tot, fr_tot = TR.sum(), FRM.sum()
        cell = {"kind": kind, "level": level, "probe": (kind, level) in PROBES, "episodes_total": E, "episodes_contributing": int((C > 0).sum()), "frames": int(fr_tot), "truth_vehicle_instances": int(tr_tot),
                "clear_matched": int(c_tot), "matched": int(m_tot), "paired_identity_matched": int(p_tot), "duplicates": int(D[n].sum()), "false_positives": int(F[n].sum()), "invalid_depth_drops": int(X[n].sum()),
                "retention": None, "c": None, "paired_identity_retention": None, "recall": None, "recall_clear_ceiling": None, "dup_per_frame": None, "fp_per_frame": None, "invalid_depth_drops_per_frame": None,
                "dup_per_clear_matched": None, "fp_per_clear_matched": None, "bootstrap_valid_resamples": None, "bootstrap_unreliable": True, "assessable": False}
        if tr_tot > 0: cell.update({"recall": m_tot / tr_tot, "recall_clear_ceiling": c_tot / tr_tot})
        if fr_tot > 0: cell.update({"dup_per_frame": D[n].sum() / fr_tot, "fp_per_frame": F[n].sum() / fr_tot, "invalid_depth_drops_per_frame": X[n].sum() / fr_tot})
        if c_tot > 0:
            ret = m_tot / c_tot
            cell.update({"retention": ret, "c": float(np.clip(1.0 - ret, 0.0, 1.0)), "paired_identity_retention": p_tot / c_tot, "dup_per_clear_matched": D[n].sum() / c_tot, "fp_per_clear_matched": F[n].sum() / c_tot})
            ep_ok = C > 0; er = M[n][ep_ok] / C[ep_ok]
            cell["per_episode_retention"] = {"n": int(ep_ok.sum()), "median": float(np.median(er)), "q25": float(np.percentile(er, 25)), "q75": float(np.percentile(er, 75)), "min": float(er.min()), "max": float(er.max())}
            cell["per_episode_paired_identity_retention"] = {"median": float(np.median(P[n][ep_ok] / C[ep_ok]))}
            fd, fq = np.asarray(frame_delta[n]), np.asarray(frame_ratio[n])
            cell["per_frame_paired"] = {"n_frames": int(fd.size), "mean_matched_delta_vs_clear": float(fd.mean()), "frac_weather_less": float((fd < 0).mean()), "frac_equal": float((fd == 0).mean()),
                                        "frac_weather_more": float((fd > 0).mean()), "median_ratio_frames_with_clear_match": float(np.median(fq)) if fq.size else None}
        if idx is not None and c_tot > 0:
            cs, ms, ps = C[idx].sum(1), M[n][idx].sum(1), P[n][idx].sum(1); trs, frs = TR[idx].sum(1), FRM[idx].sum(1); ok = cs > 0
            r = ms[ok] / cs[ok]; cb = np.clip(1.0 - r, 0.0, 1.0); nv = int(ok.sum())
            cell.update({"retention_ci95": _pct(r), "c_ci95": _pct(cb), "c_ci_half_width": float((_pct(cb)[1] - _pct(cb)[0]) / 2), "paired_identity_retention_ci95": _pct(ps[ok] / cs[ok]),
                         "bootstrap_valid_resamples": nv, "bootstrap_excluded_zero_denominator": int(BOOT_N - nv), "bootstrap_unreliable": nv < MIN_VALID_RESAMPLES})
            okf = frs > 0; okt = trs > 0
            if okt.any(): cell["recall_ci95"] = _pct(ms[okt] / trs[okt])
            if okf.any():
                for key, A_ in (("dup_per_frame_ci95", D[n]), ("fp_per_frame_ci95", F[n]), ("invalid_depth_drops_per_frame_ci95", X[n])): cell[key] = _pct(A_[idx].sum(1)[okf] / frs[okf])
            cell["assessable"] = bool(cell["c_ci_half_width"] <= CI_HALF_WIDTH_LIMIT and not cell["bootstrap_unreliable"])
        out[n] = cell
    return out


def coverage(episodes):
    if not episodes: return {"episodes": 0, "clear_matched_vehicles": 0, "episodes_contributing": 0, "clear_matched_ge_200": False, "contributing_ge_12": False, "satisfied": False}
    clear = [sum(fr["cond"]["clear:0.0"]["matched"] for fr in e["retention"]) for e in episodes]; cm, ce = int(sum(clear)), int(sum(1 for c in clear if c > 0))
    return {"episodes": len(episodes), "clear_matched_vehicles": cm, "episodes_contributing": ce, "clear_matched_ge_200": cm >= MIN_CLEAR_MATCHED, "contributing_ge_12": ce >= MIN_EPISODES,
            "satisfied": bool(cm >= MIN_CLEAR_MATCHED and ce >= MIN_EPISODES)}


def vehicle_free(episodes):
    out = {}
    for n in COND_NAMES:
        vf = [fr["cond"][n] for e in episodes for fr in e["vehicle_free"]]
        out[n] = {"frames": len(vf), "mean_false_positive_objects_n_corr": float(np.mean([c["n_corr"] for c in vf])) if vf else None,
                  "mean_valid_depth_share": float(np.mean([c["valid_depth_share"] for c in vf])) if vf else None,
                  "mean_yolo_conf": float(np.mean([c["conf_mean"] for c in vf if c.get("conf_mean") is not None])) if any(c.get("conf_mean") is not None for c in vf) else None}
    return out


def summarize(a):
    out = Path(a.out); eps = training_view(out, a.include_reserve)
    if not eps: raise SystemExit("no complete training episodes found")
    held = []
    for f in sorted((out / "episodes").glob("s*.json")):
        e = json.loads(f.read_text())
        if e["partition"] == "heldout" and e.get("status") == "complete": held.append({"seed": e["seed"], "steps": e["steps"], "stream": e["stream"]["scope"], "n_expected": e["stream"]["n_expected"], "n_valid": e["stream"]["n_valid"], "missing": e["stream"]["missing"],
                         "weather_stride": e["stream"].get("weather_stride"), "n_weather_expected": e["stream"].get("n_weather_expected"), "n_weather_valid": e["stream"].get("n_weather_valid"),
                         "n_weather_unsampled_by_design": e["stream"].get("n_weather_unsampled_by_design")})
    cells = retention_cells(eps); first20 = sorted(eps, key=lambda e: e["seed"])[:20]; cov_all, cov20 = coverage(eps), coverage(first20)
    fails = sorted(p.name for p in (out / "failures").glob("*.json")) if (out / "failures").exists() else []
    sdir = out / "summary"; sdir.mkdir(exist_ok=True); path = sdir / a.name
    if path.exists(): raise SystemExit(f"{path} exists; refusing to overwrite (choose --name)")
    doc = {"schema": "retention_camera_summary/2", "created": time.strftime("%F %T"), "definition": "retention = pooled matched(weather)/pooled matched(clear) on identical frames; c = clip(1 - retention, 0, 1)",
           "episodes": [e["seed"] for e in eps], "groups": {str(g): [e["seed"] for e in eps if e["group"] == g] for g in sorted({e["group"] for e in eps})},
           "table_c": {n: c["c"] for n, c in cells.items() if not c["probe"]}, "probe_cells": {n: c for n, c in cells.items() if c["probe"]}, "cells": cells,
           "flags": {"bootstrap_unreliable": [n for n, c in cells.items() if c["bootstrap_unreliable"]], "not_assessable": [n for n, c in cells.items() if not c["assessable"] and not c["probe"]]},
           "coverage_all_episodes": cov_all, "coverage_first_20_episodes": cov20,
           "reserve_rule_triggered": None if len(eps) < 20 else bool(not cov20["satisfied"]), "vehicle_free_analysis_only": vehicle_free(eps), "heldout_streams": held,
           "training_streams": {"clear_only_every_step": len(eps), "episodes_with_missing_frames": [e["seed"] for e in eps if e["stream"]["missing"]]}, "failure_records": fails,
           "bootstrap": {"resamples": BOOT_N, "seed": BOOT_SEED, "unit": "episode", "interval": "95% percentile", "min_valid_resamples": MIN_VALID_RESAMPLES, "zero_denominator": "excluded and counted"},
           "table_is_raw_unsmoothed": True, "probes_excluded_from_table_c": sorted(cname(*c) for c in PROBES)}
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(json.dumps(doc, indent=1)); tmp.replace(path)
    print(f"{len(eps)} training episodes | clear matched {cov_all['clear_matched_vehicles']} | contributing {cov_all['episodes_contributing']} | coverage(all) {cov_all['satisfied']} | coverage(first 20) {cov20['satisfied']} | held-out streams {len(held)}")
    for n, c in cells.items():
        f = lambda v: "  -  " if v is None else "%.3f" % v
        print("%-10s %-5s ret %s  c %s  paired %s  recall %s  boot %s%s  matched %d/%d  eps %d" % (n, "PROBE" if c["probe"] else "", f(c["retention"]), f(c["c"]), f(c["paired_identity_retention"]), f(c["recall"]),
              c["bootstrap_valid_resamples"], " UNRELIABLE" if c["bootstrap_unreliable"] else "", c["matched"], c["clear_matched"], c["episodes_contributing"]))
    print("written", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["summarize"]); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--name", default="retention_table_camera.json")
    ap.add_argument("--include-reserve", action="store_true"); a = ap.parse_args(); summarize(a)
