"""Evaluate a trained BEV behavioural-cloning checkpoint in MetaDrive."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bev_state import BEVStateAssembler
from bev_policy.model import BEVScalarPolicy
from bev_policy.metrics import action_metrics, summarise_episode_rows
from bev_policy.runtime import make_env
from bev_policy.perception import run_perception


def main_camera_frame(env):
    """Capture MetaDrive's chase-camera view for the single-window dashboard."""
    import cv2
    image = env.engine.get_sensor("main_camera").perceive(to_float=False)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def energy_proxy_metrics(speeds_mps, dt_s=0.1):
    """Flat-road, no-regeneration longitudinal energy proxy, not measured EV energy."""
    if not speeds_mps:
        return {"estimated_energy_kwh": 0.0, "estimated_wh_per_km": None}
    mass_kg, cd, area_m2, rho, crr, gravity = 1800.0, 0.23, 2.22, 1.225, 0.01, 9.81
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    accel = np.diff(np.r_[speeds[0], speeds]) / dt_s
    force = mass_kg * accel + 0.5 * rho * cd * area_m2 * speeds**2 + crr * mass_kg * gravity
    energy_j = float(np.maximum(force * speeds * dt_s, 0.0).sum())
    distance_km = float(speeds.sum() * dt_s / 1000.0)
    return {
        "estimated_energy_kwh": energy_j / 3.6e6,
        "estimated_wh_per_km": energy_j / 3600.0 / distance_km if distance_km > 1e-9 else None,
    }


def build_perception_dashboard(
    simulation_view, frames, detections, front_depth, state, action,
    weather, weather_level, step, policy_name,
):
    """Compose the simulator, policy state, five cameras, and stereo depth."""
    import cv2
    from bev_state.live_visualize import colourise_bev
    from manual_drive_stereo_yolo_weather import annotate_with_depth, depth_visualization, tile

    bev = cv2.resize(
        colourise_bev(state.bev_grid), (304, 304), interpolation=cv2.INTER_NEAREST
    )
    telemetry = np.full((304, 236, 3), 20, dtype=np.uint8)
    flat_action = np.asarray(action).reshape(-1)
    action_values = np.pad(flat_action[:2], (0, max(0, 2 - flat_action.size)))
    box_counts = [
        len(detections.get(name, ()))
        for name in ("front_left_camera", "left_camera", "right_camera", "rear_camera")
    ]
    depth = front_depth
    if depth is None:
        depth = np.zeros(frames["front_left_camera"].shape[:2], dtype=np.float32)
    valid_depth = depth[depth > 0]
    depth_coverage = 100.0 * valid_depth.size / depth.size if depth.size else 0.0
    depth_median = float(np.median(valid_depth)) if valid_depth.size else None
    lines = (
        f"Policy: {policy_name}",
        f"Step: {step}",
        f"Speed: {state.scalar_state[0]:.1f} m/s",
        f"Steer: {action_values[0]:+.2f}",
        f"Throttle/brake: {action_values[1]:+.2f}",
        f"Boxes FL/L/R/Rear: {'/'.join(map(str, box_counts))}",
        f"Stereo valid: {depth_coverage:.1f}%",
        f"Median Z: {depth_median:.1f}m" if depth_median is not None else "Median Z: no matches",
        f"Weather: {weather} ({weather_level:.2f})",
        "",
        "Q: stop evaluation",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            telemetry, line, (10, 21 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (235, 235, 235), 1, cv2.LINE_AA,
        )
    top = np.hstack((
        tile(simulation_view, "MetaDrive autonomous driving", 540, 304),
        np.hstack((bev, telemetry)),
    ))
    left = annotate_with_depth(
        frames["front_left_camera"], detections["front_left_camera"], front_depth
    )
    camera_top = np.hstack((
        tile(left, "FRONT LEFT: YOLO + stereo", 360, 270),
        tile(frames["front_right_camera"], "FRONT RIGHT: stereo partner", 360, 270),
        tile(depth_visualization(depth), "FRONT STEREO DEPTH: 1-30 m", 360, 270),
    ))
    camera_bottom = np.hstack((
        tile(annotate_with_depth(frames["left_camera"], detections["left_camera"], None),
             "LEFT: YOLO monocular", 360, 270),
        tile(annotate_with_depth(frames["right_camera"], detections["right_camera"], None),
             "RIGHT: YOLO monocular", 360, 270),
        tile(annotate_with_depth(frames["rear_camera"], detections["rear_camera"], None),
             "REAR: YOLO monocular", 360, 270),
    ))
    image = np.vstack((top, camera_top, camera_bottom))
    header = np.zeros((28, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"One-window automatic driving | {policy_name} | weathered YOLO + synchronized stereo",
        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack((header, image))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("bev_policy/checkpoints/bc_clear.pt"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-scenarios", type=int, default=10000,
                        help="MetaDrive scenario pool size; enlarge for seed-disjoint evaluation beyond training range.")
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--perception-view",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show the combined driving/perception window; defaults on with --render.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--inference-size", type=int, default=640)
    parser.add_argument("--stereo-scale", type=float, default=1.0,
                        help="Scale for front StereoSGBM matching in [0.25,1]; output depth remains camera-sized.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--weather", choices=("none", "clear", "fog", "rain", "mixed"), default="none",
                        help="Synthetic visual condition applied to the perception input.")
    parser.add_argument("--weather-level", type=float, default=0.5,
                        help="Intensity in [0,1], or maximum randomized level with --random-weather.")
    parser.add_argument("--random-weather", action="store_true",
                        help="Randomize the selected condition (or mixed conditions) and intensity per episode.")
    parser.add_argument("--visibility-range-m", type=float, default=None,
                        help="Privileged forward BEV visibility horizon in metres; not camera-inferred fog range.")
    parser.add_argument("--weather-seed", type=int, default=20260829,
                        help="Base seed for reproducible synthetic weather realizations.")
    parser.add_argument("--output-dir", type=Path, default=Path("bev_policy/outputs/weather"),
                        help="Directory for per-condition JSON and CSV results.")
    parser.add_argument("--record-video", type=Path, default=None,
                        help="Optional MP4 of the front-left camera for smooth post-run playback.")
    parser.add_argument("--record-dashboard", type=Path, default=None,
                        help="Optional MP4 of the full perception dashboard; implies --render.")
    parser.add_argument("--video-fps", type=float, default=10.0)
    args = parser.parse_args()
    show_perception = args.render if args.perception_view is None else args.perception_view
    if args.record_dashboard:
        show_perception = True
    if (
        not 0.0 <= args.weather_level <= 1.0
        or args.num_scenarios < 1
        or args.inference_size < 32
        or not 0.25 <= args.stereo_scale <= 1.0
        or not 0.0 <= args.confidence_threshold <= 1.0
    ):
        raise ValueError(
            "--weather-level and --confidence-threshold must be in [0,1], "
            "--stereo-scale in [0.25,1], and --inference-size at least 32."
        )
    if args.visibility_range_m is not None and args.visibility_range_m < 0.0:
        raise ValueError("--visibility-range-m must be greater than or equal to zero.")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    is_ppo = args.checkpoint.suffix.lower() == ".zip"
    if is_ppo:
        from stable_baselines3 import PPO
        from bev_policy.ppo import BEVFusionFeatures
        metadata_path = args.checkpoint.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"PPO normalization metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        mean = np.asarray(metadata["scalar_mean"], dtype=np.float32)
        std = np.asarray(metadata["scalar_std"], dtype=np.float32)
        policy = PPO.load(str(args.checkpoint), device=str(device))
    else:
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        policy = BEVScalarPolicy(
            in_channels=saved["in_channels"],
            bev_pool_size=int(saved.get("bev_pool_size", 1)),
        ).to(device)
        policy.load_state_dict(saved["model_state"]); policy.eval()
        mean = np.asarray(saved["scalar_mean"], dtype=np.float32)
        std = np.asarray(saved["scalar_std"], dtype=np.float32)
    from manual_drive_stereo_yolo_weather import CAMERA_RIGS, frame_from_sensor
    from yolo.yolo_utils import ensure_yolo_model
    yolo = ensure_yolo_model(
        str(device), args.yolo_model, None, args.confidence_threshold
    )
    yolo.inference_size = args.inference_size
    env, assembler = make_env(
        use_idm=False,
        render=args.render and not show_perception,
        traffic_density=args.traffic_density,
        horizon=args.max_steps,
        num_scenarios=args.num_scenarios,
        composite_view=show_perception,
    ), BEVStateAssembler()
    results = []
    writer = None
    dashboard_writer = None
    perception_window = "Automatic BEV driving + five-camera YOLO/stereo"
    stop_requested = False
    perception_window_open = False
    if show_perception:
        import cv2
        perception_cv = cv2
    try:
        if show_perception:
            perception_cv.namedWindow(perception_window, perception_cv.WINDOW_NORMAL)
            perception_window_open = True
            perception_cv.resizeWindow(perception_window, 1080, 872)
        for episode in range(args.episodes):
            seed = args.seed + episode; _, info = env.reset(seed=seed)
            weather_rng = np.random.default_rng(args.weather_seed + seed)
            weather = args.weather
            if weather == "mixed":
                weather = str(weather_rng.choice(("clear", "fog", "rain")))
            weather_level = (float(weather_rng.uniform(0.0, args.weather_level))
                             if args.random_weather else args.weather_level)
            assembler.mount_if_needed(env)
            speeds, actions, final_info = [], [], dict(info)
            box_totals = {name: 0 for name in ("front_left_camera", "left_camera", "right_camera", "rear_camera")}
            box_class_totals = {name: 0 for name in ("car", "truck", "bus", "motorcycle", "person")}
            frames_with_detections = 0
            frames_with_projected_objects = 0
            depth_coverages = []
            lateral_deviations, heading_errors = [], []
            for step in range(args.max_steps):
                raw_frames = {name: frame_from_sensor(env, name) for name in CAMERA_RIGS}
                simulation_view = main_camera_frame(env) if show_perception else None
                perception_frames, detections, weather_horizon, front_depth = run_perception(
                    raw_frames, yolo, weather, weather_level, weather_rng,
                    stereo_matcher=assembler.matcher, stereo_scale=args.stereo_scale,
                    return_depth=True,
                )
                state = assembler.assemble(
                    env=env, info=info, detections_by_camera=detections, frames=perception_frames,
                    visibility_range_m=(args.visibility_range_m if args.visibility_range_m is not None
                                        else weather_horizon),
                    front_depth_map=front_depth,
                )
                lateral_deviations.append(float(state.scalar_state[2]))
                heading_errors.append(float(state.scalar_state[3]))
                frame_boxes = sum(len(detections.get(name, ())) for name in box_totals)
                for name in box_totals:
                    box_totals[name] += len(detections.get(name, ()))
                    for detection in detections.get(name, ()):
                        label = str(detection.get("label", "unknown")).lower()
                        if label in box_class_totals:
                            box_class_totals[label] += 1
                frames_with_detections += int(frame_boxes > 0)
                frames_with_projected_objects += int(bool(state.objects))
                if state.front_depth is not None and state.front_depth.size:
                    depth_coverages.append(float(np.count_nonzero(state.front_depth > 0) / state.front_depth.size))
                if args.record_video:
                    import cv2
                    frame = state.frames["front_left_camera"]
                    if writer is None:
                        args.record_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(args.record_video), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps,
                                                 (frame.shape[1], frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video output: {args.record_video}")
                    writer.write(frame)
                normalized_scalar = ((state.scalar_state - mean) / std).astype(np.float32)
                if is_ppo:
                    action, _ = policy.predict(
                        {"bev": state.bev_grid[None].astype(np.float32),
                         "scalar": normalized_scalar},
                        deterministic=True,
                    )
                else:
                    bev = torch.from_numpy(state.bev_grid[None, None]).to(device)
                    scalar = torch.from_numpy(normalized_scalar[None]).to(device)
                    with torch.no_grad():
                        action = policy(bev, scalar).squeeze(0).cpu().numpy()
                if show_perception:
                    dashboard = build_perception_dashboard(
                        simulation_view, state.frames, detections, state.front_depth,
                        state, action, weather, weather_level, step,
                        "PPO" if is_ppo else "BC",
                    )
                    if args.record_dashboard:
                        import cv2
                        if dashboard_writer is None:
                            args.record_dashboard.parent.mkdir(parents=True, exist_ok=True)
                            dashboard_writer = cv2.VideoWriter(
                                str(args.record_dashboard), cv2.VideoWriter_fourcc(*"mp4v"),
                                args.video_fps, (dashboard.shape[1], dashboard.shape[0]),
                            )
                            if not dashboard_writer.isOpened():
                                raise RuntimeError(f"Could not open dashboard video: {args.record_dashboard}")
                        dashboard_writer.write(dashboard)
                    perception_cv.imshow(perception_window, dashboard)
                    if perception_cv.waitKey(1) & 0xFF == ord("q"):
                        stop_requested = True
                        break
                _, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
                actions.append(np.asarray(action, dtype=np.float32))
                speeds.append(float(getattr(env.agent, "speed_km_h", 0.0)) / 3.6); final_info = dict(info)
                if args.render and not show_perception:
                    env.render(text={"mode": "BEV BC evaluation", "seed": seed, "step": step})
                if terminated or truncated:
                    break
            outcome = "success" if final_info.get("arrive_dest") else "collision" if (final_info.get("crash_vehicle") or final_info.get("crash_object")) else "off_road" if final_info.get("out_of_road") else "max_step"
            row = {"seed": seed, "weather": weather, "weather_level": weather_level,
                   "weather_visibility_horizon_m": (args.visibility_range_m
                                                    if args.visibility_range_m is not None
                                                    else weather_horizon),
                   "outcome": outcome, "steps": len(speeds),
                   "mean_speed_mps": float(np.mean(speeds)) if speeds else 0.0,
                   "yolo_box_totals_by_camera": box_totals,
                   "yolo_box_totals_by_class": box_class_totals,
                   "yolo_vehicle_box_total": sum(box_class_totals[label]
                                                  for label in ("car", "truck", "bus", "motorcycle")),
                   "frames_with_any_yolo_box": frames_with_detections,
                   "frames_with_projected_objects": frames_with_projected_objects,
                   "mean_front_stereo_valid_fraction": float(np.mean(depth_coverages)) if depth_coverages else None,
                   "mean_abs_lateral_deviation_m": float(np.mean(np.abs(lateral_deviations))) if lateral_deviations else None,
                   "mean_abs_heading_error_rad": float(np.mean(np.abs(heading_errors))) if heading_errors else None,
                   **energy_proxy_metrics(speeds),
                   **action_metrics(actions)}
            results.append(row); print(row)
            if stop_requested:
                break
    finally:
        env.close()
        if writer is not None:
            writer.release()
        if dashboard_writer is not None:
            dashboard_writer.release()
        if perception_window_open:
            perception_cv.destroyWindow(perception_window)
    summary = summarise_episode_rows(results)
    report = {"config": {"checkpoint": str(args.checkpoint), "episodes": args.episodes, "seed": args.seed,
                         "max_steps": args.max_steps, "traffic_density": args.traffic_density,
                         "num_scenarios": args.num_scenarios,
                         "weather": args.weather, "weather_level": args.weather_level,
                         "yolo_model": args.yolo_model,
                         "inference_size": args.inference_size,
                         "confidence_threshold": args.confidence_threshold,
                         "stereo_scale": args.stereo_scale,
                         "random_weather": args.random_weather,
                         "visibility_range_m": args.visibility_range_m,
                         "weather_seed": args.weather_seed,
                         "energy_proxy_model": {
                             "description": "Flat-road longitudinal traction-energy proxy; not MetaDrive measured energy, battery use, or validated vehicle consumption.",
                             "dt_s": 0.1, "mass_kg": 1800.0, "drag_coefficient": 0.23,
                             "frontal_area_m2": 2.22, "air_density_kg_m3": 1.225,
                             "rolling_resistance": 0.01, "regeneration": "ignored",
                         },
                         "weather_definition": "Configurable, reproducible camera-image formation approximation: fog/rain attenuation uses a ground-plane depth prior; rain also adds exposure streaks, wet-road darkening, and lens droplets. Policy StereoSGBM depth is computed from the degraded camera pair. The associated conservative BEV horizon is an experimental intervention; MetaDrive vehicle dynamics and road friction are unchanged.",
                         "visibility_definition": "When set, visibility_range_m is a privileged controlled-visibility intervention that masks forward BEV cells; it is not camera-inferred physical fog range."},
              "episodes": results, "summary": summary}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visibility_stem = "default" if args.visibility_range_m is None else f"{args.visibility_range_m:.2f}m"
    traffic_stem = f"{args.traffic_density:.2f}"
    random_stem = "_random" if args.random_weather else ""
    stem = f"{args.weather}{random_stem}_level_{args.weather_level:.2f}_traffic_{traffic_stem}_visibility_{visibility_stem}_seed_{args.seed}_n_{args.episodes}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ["seed", "outcome"])
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary, indent=2))
    print(f"Saved structured results: {json_path} and {csv_path}")
    if args.record_video:
        print(f"Smooth playback video: {args.record_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
