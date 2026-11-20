"""Recalculate a saved PPO study from canonical per-episode speed traces.

The matching evaluation run must contain one NPZ trace per episode, written by
``bev_policy.evaluate``. Original summaries/outcomes are only read, never edited.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bev_rl.energy import EnergyParams, trace_energy


def legacy_proxy(speeds_mps: np.ndarray, dt_s: float, mass_kg: float = 1800.0) -> float:
    """Reproduce bev_policy.evaluate's former 1800 kg point-force calculation."""
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    if speeds.size < 1:
        return float("nan")
    accel = np.diff(np.r_[speeds[0], speeds]) / dt_s
    force = (mass_kg * accel + 0.5 * 1.225 * 0.23 * 2.22 * speeds**2
             + 0.01 * mass_kg * 9.81)
    work_j = float(np.maximum(force * speeds * dt_s, 0.0).sum())
    distance_km = float(speeds.sum() * dt_s / 1000.0)
    return work_j / 3600.0 / distance_km if distance_km > 1e-9 else float("nan")


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values: list[float]) -> tuple[float | None, float | None, int]:
    finite = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not finite.size:
        return None, None, 0
    return float(finite.mean()), float(finite.std(ddof=1)) if finite.size > 1 else 0.0, int(finite.size)


def standardize(original_root: Path, rerun_root: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    old_summary = _read(original_root / "paper_summary.json")
    old_conditions = old_summary["study"]["conditions"]
    new_conditions = _read(rerun_root / "paper_summary.json")["study"]["conditions"]
    if old_conditions != new_conditions:
        raise ValueError("Original and trace-capture runs use different weather-condition grids.")

    per_episode: list[dict] = []
    grouped: list[dict] = []
    compare_rows: list[dict] = []
    missing: list[str] = []
    for weather, level in old_conditions:
        label = f"{weather}_{level:.2f}"
        old_path = original_root / label / f"{weather}_level_{level:.2f}_traffic_0.10_visibility_default_seed_30000_n_20.json"
        new_path = rerun_root / label / f"{weather}_level_{level:.2f}_traffic_0.10_visibility_default_seed_30000_n_20.json"
        old_data, new_data = _read(old_path), _read(new_path)
        old_by_seed = {int(e["seed"]): e for e in old_data["episodes"]}
        new_by_seed = {int(e["seed"]): e for e in new_data["episodes"]}
        if old_by_seed.keys() != new_by_seed.keys():
            raise ValueError(f"Seed IDs differ for {label}; refusing to pair episodes.")
        config = old_data["config"]
        old_dt = float(config["energy_proxy_model"]["dt_s"])
        old_mass = float(config["energy_proxy_model"]["mass_kg"])
        condition_values = []
        for seed in sorted(old_by_seed):
            original, rerun = old_by_seed[seed], new_by_seed[seed]
            trace_name = rerun.get("energy_trace_file")
            trace_path = rerun_root / label / str(trace_name or "")
            if not trace_name or not trace_path.is_file():
                missing.append(f"{label} seed {seed}: missing trace {trace_name!r}")
                continue
            with np.load(trace_path) as trace:
                speeds = np.asarray(trace["speeds_mps"], dtype=np.float64)
                dt_s = float(trace["dt_s"])
                mass_kg = float(trace["mass_kg"])
            if speeds.size != int(rerun["steps"]) + 1:
                missing.append(f"{label} seed {seed}: trace samples do not equal steps + 1")
                continue
            if not np.isclose(dt_s, float(rerun["energy_dt_s"])):
                raise ValueError(f"Trace/report timestep mismatch for {label} seed {seed}.")

            # Historical method used post-action speeds only. The initial speed
            # sample is excluded here to reproduce its exact point-force setup.
            old_reconstructed = legacy_proxy(speeds[1:], old_dt, old_mass)
            canonical_actual = trace_energy(speeds, dt_s, EnergyParams(mass_kg=mass_kg))["wh_per_km"]
            canonical_old_mass = trace_energy(speeds, dt_s, EnergyParams(mass_kg=old_mass))["wh_per_km"]
            values = {
                "model": "ppo_bev_adaptive_safety_v1", "condition": label,
                "weather": weather, "weather_level": level, "seed": seed,
                "original_outcome": original["outcome"], "rerun_outcome": rerun["outcome"],
                "outcome_match": original["outcome"] == rerun["outcome"],
                "steps_original": original["steps"], "steps_rerun": rerun["steps"],
                "mass_kg_used": mass_kg, "dt_s_used": dt_s,
                "original_legacy_wh_per_km": original.get("estimated_wh_per_km"),
                "legacy_recomputed_on_rerun_trace_wh_per_km": old_reconstructed,
                "canonical_wh_per_km_at_1800kg": canonical_old_mass,
                "canonical_wh_per_km_standardized": canonical_actual,
                "mass_effect_wh_per_km": canonical_actual - canonical_old_mass,
                "integration_and_speed_sampling_effect_wh_per_km": canonical_old_mass - old_reconstructed,
                "total_change_vs_original_wh_per_km": canonical_actual - float(original["estimated_wh_per_km"]),
                "legacy_reproduction_error_wh_per_km": old_reconstructed - float(original["estimated_wh_per_km"]),
                "energy_trace_file": trace_name,
            }
            per_episode.append(values)
            condition_values.append(values)

        all_mean, all_sd, n_all = _mean_sd([float(r["canonical_wh_per_km_standardized"]) for r in condition_values])
        successful = [r for r in condition_values if r["original_outcome"] == "success"]
        success_mean, success_sd, n_success = _mean_sd([float(r["canonical_wh_per_km_standardized"]) for r in successful])
        original_vals = [float(r["original_legacy_wh_per_km"]) for r in condition_values]
        old_mean, old_sd, _ = _mean_sd(original_vals)
        compare_rows.append({"condition": label, "old_mean": old_mean, "standardized_mean": all_mean,
                             "all_episode_change": all_mean - old_mean if all_mean is not None and old_mean is not None else None,
                             "n": n_all})
        grouped.append({"model": "ppo_bev_adaptive_safety_v1", "weather": weather,
                        "weather_level": level, "condition": label,
                        "all_episode_n": n_all, "all_episode_mean_wh_per_km": all_mean,
                        "all_episode_sd_wh_per_km": all_sd,
                        "successful_episode_n": n_success,
                        "successful_episode_mean_wh_per_km": success_mean,
                        "successful_episode_sd_wh_per_km": success_sd,
                        "original_legacy_mean_wh_per_km": old_mean,
                        "original_legacy_sd_wh_per_km": old_sd})

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_csv = output_dir / "ppo_energy_standardized_per_episode.csv"
    summary_csv = output_dir / "ppo_energy_standardized_summary.csv"
    for path, rows in ((episode_csv, per_episode), (summary_csv, grouped)):
        if rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)

    report_path = output_dir / "energy_standardization_validation.md"
    report = [
        "# PPO energy standardization and validation", "",
        "The canonical value is estimated mechanical traction work per kilometre, not measured battery energy.",
        "It uses `bev_rl.energy.trace_energy`, the simulator-reported mass, the trace's configured decision timestep,",
        "closed-form work for linear speed changes, and clips negative work at each step (no regeneration).", "",
        "Original outcomes and success classifications are preserved. Successful-only summaries select episodes whose",
        "original saved outcome was `success`; rerun outcome differences are reported and do not replace it.", "",
        "| Condition | Original proxy mean ± SD | Standardized all episodes mean ± SD | Success-only mean ± SD | n success |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in grouped:
        report.append(f"| {row['condition']} | {row['original_legacy_mean_wh_per_km']:.2f} ± {row['original_legacy_sd_wh_per_km']:.2f} | "
                      f"{row['all_episode_mean_wh_per_km']:.2f} ± {row['all_episode_sd_wh_per_km']:.2f} | "
                      f"{row['successful_episode_mean_wh_per_km']:.2f} ± {row['successful_episode_sd_wh_per_km']:.2f} | "
                      f"{row['successful_episode_n']} |")
    mismatches = [r for r in per_episode if not r["outcome_match"]]
    old_rank = [r["condition"] for r in sorted(compare_rows, key=lambda x: x["old_mean"] or float("inf"))]
    new_rank = [r["condition"] for r in sorted(compare_rows, key=lambda x: x["standardized_mean"] or float("inf"))]
    mass_values = [float(r["mass_effect_wh_per_km"]) for r in per_episode]
    method_values = [float(r["integration_and_speed_sampling_effect_wh_per_km"]) for r in per_episode]
    report += ["", "## Controlled decomposition", "",
               "For each rerun trace, the mass effect is canonical live mass minus canonical 1800 kg;",
               "the integration and speed-sampling effect is canonical 1800 kg minus the historical calculation replayed on the same post-action speed samples.",
               "These controlled intermediate comparisons isolate mass and methodology on the rerun trace. They do not remove possible differences between the original run and its seeded replay.", "",
               f"Recorded simulator masses (kg): {sorted(set(round(float(r['mass_kg_used']), 6) for r in per_episode))}.",
               f"Recorded decision intervals (s): {sorted(set(round(float(r['dt_s_used']), 9) for r in per_episode))}.",
               f"Mean isolated mass effect: {np.mean(mass_values):+.2f} Wh/km; mean integration/sampling effect: {np.mean(method_values):+.2f} Wh/km.",
               f"Weather-condition ranking by mean Wh/km, lowest to highest: old {', '.join(old_rank)}; standardized {', '.join(new_rank)}. Ranking changed: {old_rank != new_rank}.", "",
               f"Episode outcome mismatches between the preserved original and trace-capture rerun: {len(mismatches)} / {len(per_episode)}.",
               f"Episodes with insufficient/missing traces: {len(missing)}."]
    for item in missing:
        report.append(f"- {item}")
    report += ["", "| Condition | Mean original | Mean standardized | Change |", "|---|---:|---:|---:|"]
    for r in compare_rows:
        report.append(f"| {r['condition']} | {r['old_mean']:.2f} | {r['standardized_mean']:.2f} | {r['all_episode_change']:+.2f} |")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return episode_csv, summary_csv, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = standardize(args.original_root, args.trace_root, args.output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
