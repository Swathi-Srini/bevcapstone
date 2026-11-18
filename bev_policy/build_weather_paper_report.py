"""Build readable paper tables from the completed weather study JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "bev_policy" / "outputs" / "ppo_weather_generalization_20_stereo05"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ms(mean: float | None, sd: float | None, digits: int = 2) -> str:
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f} ± {sd or 0:.{digits}f}"


def main() -> int:
    status = json.loads((STUDY_DIR / "study_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "complete":
        raise SystemExit("Study is not complete; no final paper table was written.")
    data = json.loads((STUDY_DIR / "paper_summary.json").read_text(encoding="utf-8"))
    rows = data["conditions"]
    expected = {"clear_0.00", "fog_0.25", "fog_0.75", "rain_0.25", "rain_0.75"}
    observed = {row["condition"] for row in rows}
    if observed != expected or any(row["episodes"] != 20 for row in rows):
        raise SystemExit(f"Unexpected condition/episode set: {observed}")

    lines = [
        "# PPO weather generalization study",
        "",
        "## Experimental setup",
        "",
        "BC-initialized safety-shaped PPO checkpoint; 20 paired held-out seeds per condition "
        "(30000–30019), MetaDrive traffic density 0.10, 500-step limit, YOLOv8n at 320 px "
        "and confidence 0.25, stereo matching at 0.5 scale. The condition uses synthetic "
        "camera weather corruption; road friction and vehicle dynamics are unchanged. The "
        "BEV builder receives the synthetic visibility horizon associated with the weather "
        "condition, so this is not a camera-only weather-estimation evaluation.",
        "",
        "## Driving outcomes",
        "",
        "Success intervals are 95% Wilson intervals. Continuous values are mean ± SD across episodes.",
        "",
        "| Weather | Success | Collision | Off road | Max steps | Speed (m/s) | Lateral error (m) | Heading error (rad) | Estimated energy (Wh/km) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lo, hi = row["success_wilson_95_ci"]
        lines.append(
            f"| {row['condition']} | {row['successes']}/20 ({pct(row['success_rate'])}; "
            f"95% CI {pct(lo)}–{pct(hi)}) | {row['collisions']}/20 ({pct(row['collision_rate'])}) | "
            f"{row['off_road']} | {row['max_step']} | "
            f"{ms(row['mean_speed_mps'], row['sd_episode_mean_speed_mps'])} | "
            f"{ms(row['mean_abs_lateral_deviation_m'], row['sd_episode_mean_abs_lateral_deviation_m'])} | "
            f"{ms(row['mean_abs_heading_error_rad'], row['sd_episode_mean_abs_heading_error_rad'], 3)} | "
            f"{ms(row['mean_estimated_wh_per_km'], row['sd_episode_estimated_wh_per_km'])} |"
        )
    lines += [
        "",
        "## YOLO detection-instance ablation",
        "",
        "Counts are raw predicted box instances accumulated over episode frames, not unique tracked vehicles. "
        "No annotated ground truth is available, so these values cannot be interpreted as detection accuracy, "
        "precision, or recall. Frame-normalized values account for different episode lengths.",
        "",
        "| Weather | Vehicle boxes / frame | Vehicle box instances | Car | Truck | Bus | Motorcycle | Person | Frames with any box | Mean stereo valid fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cls = row["yolo_box_totals_by_class"]
        lines.append(
            f"| {row['condition']} | {row['mean_yolo_vehicle_boxes_per_frame']:.3f} | "
            f"{row['yolo_vehicle_box_total']} | {cls['car']} | {cls['truck']} | {cls['bus']} | "
            f"{cls['motorcycle']} | {cls['person']} | "
            f"{pct(row['frames_with_any_yolo_box_fraction'])} | "
            f"{row['mean_front_stereo_valid_fraction']:.4f} |"
        )
    lines += [
        "",
        "## YOLO box instances by camera",
        "",
        "| Weather | Front left | Left | Right | Rear |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        cams = row["yolo_box_totals_by_camera"]
        lines.append(
            f"| {row['condition']} | {cams['front_left_camera']} | {cams['left_camera']} | "
            f"{cams['right_camera']} | {cams['rear_camera']} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- This is a simulator pilot and evaluates a checkpoint trained for only 2,048 PPO timesteps; it does not establish convergence or deployment safety.",
        "- The seeds are held out from the training scenario-index range, but map/scenario diversity is limited to MetaDrive's generated scenario pool.",
        "- Synthetic weather changes camera pixels and supplies a condition-derived visibility horizon to BEV construction; it does not simulate rain/fog effects on tire-road dynamics.",
        "- Energy is an idealized flat-road tractive-work proxy, not measured battery energy; regeneration, grade, and drivetrain efficiency are omitted.",
        "- YOLO counts are predictions and can include repeated observations of the same vehicle across frames.",
        "",
        f"Episode-level source: `{data['study']['checkpoint']}` and per-condition JSON/CSV files in `{STUDY_DIR}`.",
        "",
    ]
    output = STUDY_DIR / "paper_table.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
