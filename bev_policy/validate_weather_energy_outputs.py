"""Independently validate saved weather-batch energy traces and summarize them."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from bev_rl.energy import EnergyParams, trace_energy


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    data = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not data.size:
        return None, None
    return float(data.mean()), float(data.std(ddof=1)) if data.size > 1 else 0.0


def validate(input_root: Path, expected_episodes: int = 10) -> bool:
    episode_rows: list[dict] = []
    summary_rows: list[dict] = []
    checks: list[str] = []
    failed = False
    for condition_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        reports = sorted(condition_dir.glob("*.json"))
        if not reports:
            continue
        # The batch creates one report JSON per weather condition.
        report_path = reports[-1]
        report = json.loads(report_path.read_text(encoding="utf-8"))
        config = report["config"]
        episodes = report.get("episodes", [])
        if len(episodes) != expected_episodes:
            failed = True
            checks.append(f"FAIL {condition_dir.name}: expected {expected_episodes} episodes, found {len(episodes)}")
        values_all: list[float] = []
        values_success: list[float] = []
        mismatch_count = 0
        masses, timesteps = set(), set()
        outcomes: dict[str, int] = {}
        for episode in episodes:
            outcome = str(episode["outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            trace_name = episode.get("energy_trace_file")
            if not trace_name:
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode.get('seed')}: trace filename missing")
                continue
            trace_path = condition_dir / trace_name
            if not trace_path.is_file():
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode.get('seed')}: trace missing ({trace_name})")
                continue
            with np.load(trace_path) as saved:
                speeds = np.asarray(saved["speeds_mps"], dtype=np.float64)
                dt_s = float(saved["dt_s"])
                mass_kg = float(saved["mass_kg"])
            masses.add(round(mass_kg, 8)); timesteps.add(round(dt_s, 9))
            if len(speeds) != int(episode["steps"]) + 1:
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode['seed']}: sample count != steps + 1")
                continue
            if not np.isclose(dt_s, float(episode["energy_dt_s"]), rtol=0.0, atol=1e-12):
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode['seed']}: trace and report timestep differ")
                continue
            energy = trace_energy(speeds, dt_s, EnergyParams(mass_kg=mass_kg))
            stored_whkm = float(episode["estimated_wh_per_km"])
            stored_wh = float(episode["estimated_energy_kwh"]) * 1000.0
            err_whkm = abs(energy["wh_per_km"] - stored_whkm)
            err_wh = abs(energy["traction_pos_wh"] - stored_wh)
            passed = err_whkm <= 1e-8 and err_wh <= 1e-8
            mismatch_count += int(not passed)
            if not passed:
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode['seed']}: saved metric differs from canonical trace recomputation")
            if not np.isclose(mass_kg, 1100.0, rtol=0.0, atol=1e-6):
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode['seed']}: simulator mass is {mass_kg:g} kg, expected 1100 kg")
            if not np.isclose(dt_s, 0.1, rtol=0.0, atol=1e-9):
                failed = True
                checks.append(f"FAIL {condition_dir.name} seed {episode['seed']}: decision timestep is {dt_s:g} s, expected 0.1 s")
            row = {
                "condition": condition_dir.name, "model": Path(config["checkpoint"]).stem,
                "seed": int(episode["seed"]), "weather": episode["weather"],
                "weather_level": float(episode["weather_level"]), "outcome": outcome,
                "success": outcome == "success", "steps": int(episode["steps"]),
                "mass_kg": mass_kg, "dt_s": dt_s,
                "canonical_energy_wh": energy["traction_pos_wh"],
                "canonical_wh_per_km": energy["wh_per_km"],
                "stored_energy_wh": stored_wh, "stored_wh_per_km": stored_whkm,
                "recalculation_abs_error_wh": err_wh,
                "recalculation_abs_error_wh_per_km": err_whkm,
                "validation_pass": passed and np.isclose(mass_kg, 1100.0) and np.isclose(dt_s, 0.1),
                "speed_trace_file": trace_name,
            }
            episode_rows.append(row)
            values_all.append(energy["wh_per_km"])
            if outcome == "success":
                values_success.append(energy["wh_per_km"])
        all_mean, all_sd = summarize(values_all)
        success_mean, success_sd = summarize(values_success)
        summary_rows.append({
            "model": Path(config["checkpoint"]).stem,
            "condition": condition_dir.name,
            "weather": config["weather"], "weather_level": float(config["weather_level"]),
            "episodes": len(episodes), "successes": outcomes.get("success", 0),
            "collisions": outcomes.get("collision", 0), "off_road": outcomes.get("off_road", 0),
            "max_step": outcomes.get("max_step", 0),
            "all_episode_mean_wh_per_km": all_mean, "all_episode_sd_wh_per_km": all_sd,
            "successful_episode_n": len(values_success),
            "successful_episode_mean_wh_per_km": success_mean,
            "successful_episode_sd_wh_per_km": success_sd,
            "simulator_masses_kg": ";".join(map(str, sorted(masses))),
            "decision_timesteps_s": ";".join(map(str, sorted(timesteps))),
            "trace_metric_mismatches": mismatch_count,
        })
        checks.append(f"{'PASS' if len(episodes) == expected_episodes and mismatch_count == 0 and masses == {1100.0} and timesteps == {0.1} else 'CHECK'} {condition_dir.name}: n={len(episodes)}, outcomes={outcomes}, masses={sorted(masses)}, dt={sorted(timesteps)}")

    if not summary_rows:
        raise FileNotFoundError(f"No per-condition JSON reports found under {input_root}")
    with (input_root / "energy_validation_per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader(); writer.writerows(episode_rows)
    with (input_root / "energy_validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    lines = [
        "# Weather-batch energy validation", "",
        "Wh/km is estimated mechanical traction energy, not measured battery consumption. Each saved speed trace was recalculated with `bev_rl.energy.trace_energy`, using the saved live simulator mass and decision timestep.",
        "",
        "| Model | Weather | Episodes | Success | Collision | Off road | Max step | All episodes Wh/km | Successful only Wh/km | Mass (kg) | dt (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model']} | {row['condition']} | {row['episodes']} | {row['successes']} | {row['collisions']} | {row['off_road']} | {row['max_step']} | "
                     f"{row['all_episode_mean_wh_per_km']:.2f} ± {row['all_episode_sd_wh_per_km']:.2f} | "
                     f"{(row['successful_episode_mean_wh_per_km'] if row['successful_episode_mean_wh_per_km'] is not None else float('nan')):.2f} ± "
                     f"{(row['successful_episode_sd_wh_per_km'] if row['successful_episode_sd_wh_per_km'] is not None else float('nan')):.2f} | "
                     f"{row['simulator_masses_kg']} | {row['decision_timesteps_s']} |")
    lines += ["", "## Checks", "", *[f"- {item}" for item in checks], "",
              "Outcomes are read from the evaluator report and are not recalculated or changed."]
    (input_root / "energy_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved energy validation files under {input_root}")
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=10)
    args = parser.parse_args()
    return 0 if validate(args.input_root, args.expected_episodes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
