"""Run a paired 20-seed PPO weather study and aggregate paper-ready metrics."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np

from bev_policy.evaluate import main as evaluate_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "bev_policy" / "outputs" / "ppo_weather_generalization_20_stereo05"
SEED_START = 30000  # outside the 10..10009 scenario-index pool used for training
SCENARIO_COUNT = SEED_START + 50
N_EPISODES = 20
CONDITIONS = (
    ("clear", 0.0),
    ("fog", 0.25),
    ("fog", 0.75),
    ("rain", 0.25),
    ("rain", 0.75),
)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def condition_report(weather: str, level: float, report_path: Path) -> dict:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    n = len(episodes)
    outcomes = {name: sum(row["outcome"] == name for row in episodes)
                for name in ("success", "collision", "off_road", "max_step")}
    ci = wilson_interval(outcomes["success"], n)
    camera_names = ("front_left_camera", "left_camera", "right_camera", "rear_camera")
    total_frames = sum(row["steps"] for row in episodes)
    box_totals = {
        camera: sum(row["yolo_box_totals_by_camera"][camera] for row in episodes)
        for camera in camera_names
    }
    class_names = ("car", "truck", "bus", "motorcycle", "person")
    class_totals = {
        label: sum(row["yolo_box_totals_by_class"][label] for row in episodes)
        for label in class_names
    }
    vehicle_box_total = sum(class_totals[label] for label in ("car", "truck", "bus", "motorcycle"))

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in episodes if row.get(field) is not None]

    def mean_std(field: str) -> tuple[float | None, float | None]:
        sample = values(field)
        if not sample:
            return None, None
        return float(np.mean(sample)), float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0

    speed_mean, speed_std = mean_std("mean_speed_mps")
    energy_mean, energy_std = mean_std("estimated_wh_per_km")
    success_energy = [float(row["estimated_wh_per_km"]) for row in episodes
                      if row["outcome"] == "success" and row.get("estimated_wh_per_km") is not None]
    lateral_mean, lateral_std = mean_std("mean_abs_lateral_deviation_m")
    heading_mean, heading_std = mean_std("mean_abs_heading_error_rad")
    depth_mean, depth_std = mean_std("mean_front_stereo_valid_fraction")
    det_frames = sum(row["frames_with_any_yolo_box"] for row in episodes)
    projected_frames = sum(row["frames_with_projected_objects"] for row in episodes)
    return {
        "weather": weather, "intensity": level,
        "episodes": n, "seed_start": SEED_START, "seed_end": SEED_START + n - 1,
        "successes": outcomes["success"], "success_rate": outcomes["success"] / n if n else 0.0,
        "success_wilson_95_ci": list(ci),
        "collisions": outcomes["collision"], "collision_rate": outcomes["collision"] / n if n else 0.0,
        "off_road": outcomes["off_road"], "off_road_rate": outcomes["off_road"] / n if n else 0.0,
        "max_step": outcomes["max_step"],
        "mean_speed_mps": speed_mean, "sd_episode_mean_speed_mps": speed_std,
        "mean_estimated_wh_per_km": energy_mean, "sd_episode_estimated_wh_per_km": energy_std,
        "mean_success_estimated_wh_per_km": float(np.mean(success_energy)) if success_energy else None,
        "sd_success_estimated_wh_per_km": float(np.std(success_energy, ddof=1)) if len(success_energy) > 1 else (0.0 if success_energy else None),
        "n_success_energy": len(success_energy),
        "mean_abs_lateral_deviation_m": lateral_mean,
        "sd_episode_mean_abs_lateral_deviation_m": lateral_std,
        "mean_abs_heading_error_rad": heading_mean,
        "sd_episode_mean_abs_heading_error_rad": heading_std,
        "mean_front_stereo_valid_fraction": depth_mean,
        "sd_episode_front_stereo_valid_fraction": depth_std,
        "total_decision_frames": total_frames,
        "yolo_box_totals_by_camera": box_totals,
        "yolo_box_totals_by_class": class_totals,
        "yolo_vehicle_box_total": vehicle_box_total,
        "mean_yolo_vehicle_boxes_per_frame": vehicle_box_total / total_frames if total_frames else 0.0,
        "mean_yolo_vehicle_boxes_per_episode": vehicle_box_total / n if n else 0.0,
        "mean_yolo_boxes_per_frame_by_camera": {
            camera: (box_totals[camera] / total_frames if total_frames else 0.0)
            for camera in camera_names
        },
        "frames_with_any_yolo_box_fraction": det_frames / total_frames if total_frames else 0.0,
        "frames_with_projected_bev_object_fraction": projected_frames / total_frames if total_frames else 0.0,
        "condition_report": str(report_path.relative_to(PROJECT_ROOT)),
        "episode_rows": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Write this run separately; defaults to the historical study directory.")
    parser.add_argument("--energy-standardized", action="store_true",
                        help="Label outputs as canonical bev_rl.energy results; use a new output root.")
    args = parser.parse_args()
    output_root = args.output_root
    if args.energy_standardized and output_root.resolve() == OUTPUT_ROOT.resolve():
        raise SystemExit("--energy-standardized requires a separate --output-root; historical files are protected.")
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "study_status.json"
    complete: list[dict] = []
    status = {"state": "running", "requested_episodes_per_condition": N_EPISODES,
              "seed_range": [SEED_START, SEED_START + N_EPISODES - 1],
              "traffic_density": 0.10, "inference_size": 320,
              "confidence_threshold": 0.25, "stereo_scale": 0.5,
              "conditions": []}
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    for weather, level in CONDITIONS:
        condition_name = f"{weather}_{level:.2f}"
        output_dir = output_root / condition_name
        report_path = output_dir / (
            f"{weather}_level_{level:.2f}_traffic_0.10_visibility_default_"
            f"seed_{SEED_START}_n_{N_EPISODES}.json"
        )
        status["current_condition"] = condition_name
        status["completed_conditions"] = [row["condition"] for row in complete]
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        sys.argv = [
            "bev_policy.evaluate",
            "--checkpoint", str(PROJECT_ROOT / "bev_policy" / "checkpoints" / "ppo_bev_adaptive_safety_v1.zip"),
            "--episodes", str(N_EPISODES), "--max-steps", "500",
            "--seed", str(SEED_START), "--num-scenarios", str(SCENARIO_COUNT),
            "--traffic-density", "0.10", "--weather", weather,
            "--weather-level", str(level), "--inference-size", "320",
            "--confidence-threshold", "0.25", "--stereo-scale", "0.5",
            "--output-dir", str(output_dir),
        ]
        result = evaluate_main()
        if result != 0 or not report_path.exists():
            raise RuntimeError(f"Evaluation failed to produce report for {condition_name}")
        row = condition_report(weather, level, report_path)
        row["condition"] = condition_name
        complete.append(row)
        status["conditions"] = complete
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"STUDY CONDITION COMPLETE {condition_name}: "
              f"success {row['successes']}/{row['episodes']}, "
              f"collision {row['collisions']}/{row['episodes']}, "
              f"YOLO vehicles {row['yolo_box_totals_by_class']}", flush=True)

    status["state"] = "complete"
    status.pop("current_condition", None)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    rows = [{key: value for key, value in row.items() if key != "episode_rows"}
            for row in complete]
    (output_root / "paper_summary.json").write_text(json.dumps({
        "study": {
            "policy": "BEV BC-initialized safety-shaped PPO; 2048 PPO training steps",
            "checkpoint": "bev_policy/checkpoints/ppo_bev_adaptive_safety_v1.zip",
            "conditions": [list(condition) for condition in CONDITIONS],
            "episodes_per_condition": N_EPISODES,
            "seed_range": [SEED_START, SEED_START + N_EPISODES - 1],
            "scenario_pool_size": SCENARIO_COUNT,
            "traffic_density": 0.10,
            "yolo_model": "yolov8n.pt", "inference_size": 320,
            "confidence_threshold": 0.25, "stereo_scale": 0.5,
            "weather": "synthetic camera image corruption; MetaDrive friction/dynamics unchanged",
            "energy_methodology": ("bev_rl.energy.trace_energy; mass read from the MetaDrive vehicle; per-episode speed trace and decision dt saved; closed-form linear-speed work with negative work clipped per step; no regeneration. Estimated mechanical traction energy, not battery consumption."
                                  if args.energy_standardized else "See each episode report for energy methodology."),
            "limitations": [
                "This is a simulator pilot, not real-road validation.",
                "YOLO box counts are predictions, not precision/recall; no annotated truth set is available.",
                "Energy is estimated mechanical traction energy, not measured battery consumption.",
                "The PPO checkpoint was trained for only 2048 timesteps; a 100-episode evaluation does not cure short training.",
            ],
        },
        "conditions": complete,
    }, indent=2), encoding="utf-8")
    import csv
    summary_csv = output_root / "paper_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.energy_standardized:
        from bev_policy.standardize_ppo_energy import standardize
        episode_csv, energy_summary_csv, validation_report = standardize(
            original_root=PROJECT_ROOT / "bev_policy" / "outputs" / "ppo_weather_generalization_20_stereo05",
            rerun_root=output_root,
            output_dir=output_root,
        )
        print(f"Saved standardized energy episodes: {episode_csv}", flush=True)
        print(f"Saved standardized energy summary: {energy_summary_csv}", flush=True)
        print(f"Saved energy validation: {validation_report}", flush=True)
    print(f"Saved 100-episode study: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
