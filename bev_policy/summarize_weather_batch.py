"""Build paper tables from completed weather-batch episode reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def mean_sd(rows: list[dict], key: str) -> tuple[float | None, float | None, int]:
    values = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]
    if not values:
        return None, None, 0
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, len(values)


def f(value: float | None, digits: int = 2) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def build(root: Path) -> tuple[Path, Path]:
    energy_path = root / "energy_validation_summary.csv"
    with energy_path.open(newline="", encoding="utf-8") as handle:
        energy_summary = {row["condition"]: row for row in csv.DictReader(handle)}

    rows: list[dict] = []
    for condition_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if condition_dir.name not in energy_summary:
            continue
        reports = list(condition_dir.glob("*.json"))
        if len(reports) != 1:
            raise ValueError(f"Expected one completed episode report in {condition_dir}; found {len(reports)}")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        eps = report["episodes"]
        cfg = report["config"]
        e = energy_summary[condition_dir.name]
        if len(eps) != int(e["episodes"]):
            raise ValueError(f"Episode count mismatch for {condition_dir.name}")
        outcomes = [row["outcome"] for row in eps]
        success = outcomes.count("success")
        lo, hi = wilson(success, len(eps))
        speed_mean, speed_sd, speed_n = mean_sd(eps, "mean_speed_mps")
        lateral_mean, lateral_sd, _ = mean_sd(eps, "mean_abs_lateral_deviation_m")
        heading_mean, heading_sd, _ = mean_sd(eps, "mean_abs_heading_error_rad")
        vehicle_boxes = [int(row["yolo_vehicle_box_total"]) for row in eps]
        depth_vals = [float(row["mean_front_stereo_valid_fraction"])
                      for row in eps if row.get("mean_front_stereo_valid_fraction") is not None]
        rows.append({
            "model": Path(cfg["checkpoint"]).stem,
            "traffic_density": float(cfg["traffic_density"]),
            "weather": cfg["weather"], "weather_level": float(cfg["weather_level"]),
            "condition": condition_dir.name, "episodes": len(eps),
            "successes": success, "success_rate": success / len(eps),
            "success_wilson_95_low": lo, "success_wilson_95_high": hi,
            "collisions": outcomes.count("collision"), "off_road": outcomes.count("off_road"),
            "max_step": outcomes.count("max_step"),
            "mean_speed_mps": speed_mean, "sd_episode_mean_speed_mps": speed_sd,
            "mean_abs_lateral_m": lateral_mean, "sd_episode_mean_abs_lateral_m": lateral_sd,
            "mean_abs_heading_rad": heading_mean, "sd_episode_mean_abs_heading_rad": heading_sd,
            "all_episode_wh_per_km_mean": float(e["all_episode_mean_wh_per_km"]),
            "all_episode_wh_per_km_sd": float(e["all_episode_sd_wh_per_km"]),
            "successful_episode_wh_per_km_n": int(e["successful_episode_n"]),
            "successful_episode_wh_per_km_mean": float(e["successful_episode_mean_wh_per_km"]),
            "successful_episode_wh_per_km_sd": float(e["successful_episode_sd_wh_per_km"]),
            "mean_yolo_vehicle_box_instances_per_episode": float(np.mean(vehicle_boxes)),
            "episodes_with_zero_yolo_vehicle_boxes": sum(count == 0 for count in vehicle_boxes),
            "mean_stereo_valid_fraction": float(np.mean(depth_vals)) if depth_vals else None,
            "episodes_with_stereo_metric": len(depth_vals),
            "mass_kg": e["simulator_masses_kg"], "dt_s": e["decision_timesteps_s"],
            "energy_trace_mismatches": int(e["trace_metric_mismatches"]),
            "max_steps": int(cfg["max_steps"]),
        })
    if len(rows) != 5:
        raise ValueError(f"Expected completed reports for five conditions; found {len(rows)}")

    csv_path = root / "paper_condition_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    lines = [
        "# PPO weather evaluation: paper summary", "",
        f"Checkpoint: `{rows[0]['model']}`. Traffic density: {rows[0]['traffic_density']:.3f}. "
        f"Episodes per weather: {rows[0]['episodes']}. Step cap: {rows[0]['max_steps']}.",
        "Weather is synthetic image corruption with a condition-derived visibility horizon; MetaDrive friction and vehicle dynamics are unchanged.",
        "Energy is estimated mechanical traction energy (not battery use), recalculated using `bev_rl.energy.trace_energy`; mass and timestep are read from the run traces.",
        "Success intervals are 95% Wilson intervals. Continuous statistics are mean ± sample SD across episode-level values.", "",
        "| Weather | Success (95% Wilson CI) | Collision | Off road | Max step | Speed (m/s) | Energy all episodes (Wh/km) | Energy successful only (Wh/km) | YOLO vehicle boxes/episode* | Zero-box episodes | Stereo valid fraction (n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['successes']}/{r['episodes']} ({r['success_rate']:.1%}; {100*r['success_wilson_95_low']:.1f}–{100*r['success_wilson_95_high']:.1f}%) "
            f"| {r['collisions']}/{r['episodes']} | {r['off_road']}/{r['episodes']} | {r['max_step']}/{r['episodes']} "
            f"| {f(r['mean_speed_mps'])} ± {f(r['sd_episode_mean_speed_mps'])} "
            f"| {f(r['all_episode_wh_per_km_mean'])} ± {f(r['all_episode_wh_per_km_sd'])} "
            f"| {f(r['successful_episode_wh_per_km_mean'])} ± {f(r['successful_episode_wh_per_km_sd'])} (n={r['successful_episode_wh_per_km_n']}) "
            f"| {f(r['mean_yolo_vehicle_box_instances_per_episode'], 1)} | {r['episodes_with_zero_yolo_vehicle_boxes']}/{r['episodes']} "
            f"| {f(r['mean_stereo_valid_fraction'], 4)} (n={r['episodes_with_stereo_metric']}) |"
        )
    lines += [
        "", "*YOLO box instances count repeated detections across frames; they are not unique vehicles or detection accuracy.",
        "", "## Energy validation", "",
        "All episodes were checked against their saved speed trace with canonical `bev_rl.energy.trace_energy`. "
        f"Recorded simulator mass: {rows[0]['mass_kg']} kg; decision interval: {rows[0]['dt_s']} s. "
        f"Total trace-metric mismatches: {sum(r['energy_trace_mismatches'] for r in rows)}.",
        "", "## Outcome counts", "",
        f"Across all conditions: {sum(r['successes'] for r in rows)}/100 successes, "
        f"{sum(r['collisions'] for r in rows)} collisions, {sum(r['off_road'] for r in rows)} off-road, "
        f"{sum(r['max_step'] for r in rows)} max-step terminations.",
        "", "The per-episode original outcome labels were retained; this table does not change classifications.",
    ]
    md_path = root / "paper_condition_results.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()
    for path in build(args.input_root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
