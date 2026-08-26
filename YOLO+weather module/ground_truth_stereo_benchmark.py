"""Benchmark front StereoSGBM output against MetaDrive traffic-object poses.

This evaluates the active ``manual_drive_stereo_yolo_weather`` camera rig. It
does not use YOLO box size. Each traffic vehicle's true MetaDrive pose and
dimensions are used to project its visible longitudinal surface through the
actually-mounted front-left camera; SGBM is sampled at that projected pixel.

The comparison is limited to the module's declared 1--30 m operating range.
It is an auditable simulator pose/surface reference, not a replacement for a
per-pixel depth-renderer benchmark.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np
from panda3d.core import Point2, Point3

from manual_drive_stereo_yolo_weather import (
    CAMERA_RIGS,
    CAM_H,
    CAM_W,
    FOCAL_LENGTH_PX,
    build_stereo_matcher,
    frame_from_sensor,
    front_stereo_depth,
    mount_camera_rig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare active StereoSGBM depth with MetaDrive object-pose references.")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--traffic-density", type=float, default=0.5)
    parser.add_argument("--start-seed", type=int, default=0, help="Fixed MetaDrive scenario seed for reproducible results.")
    parser.add_argument("--output-dir", type=Path, default=Path("integrated_output/ground_truth_benchmark"))
    return parser.parse_args()


def project_world_point(camera, render, world_point: Point3) -> tuple[float, float, float] | None:
    """Return pixel x/y and positive camera-forward range for a world point."""
    point_camera = camera.getRelativePoint(render, world_point)
    if point_camera.y <= 0.0:
        return None
    screen = Point2()
    if not camera.node().getLens().project(point_camera, screen):
        return None
    u = (screen.x + 1.0) * 0.5 * CAM_W
    v = (1.0 - screen.y) * 0.5 * CAM_H
    if not (0 <= u < CAM_W and 0 <= v < CAM_H):
        return None
    return float(u), float(v), float(point_camera.y)


def sample_depth(depth: np.ndarray, u: float, v: float, radius: int = 12) -> float | None:
    """Robustly sample valid SGBM values around a projected object centre."""
    x, y = int(round(u)), int(round(v))
    patch = depth[max(0, y - radius):min(depth.shape[0], y + radius + 1),
                  max(0, x - radius):min(depth.shape[1], x + radius + 1)]
    values = patch[patch > 0]
    return None if values.size == 0 else float(np.median(values))


def visible_surface_point(vehicle, camera, render) -> tuple[Point3, float] | None:
    """Return the vehicle longitudinal surface that faces the front-left camera."""
    centre = Point3(float(vehicle.position[0]), float(vehicle.position[1]), 0.8)
    camera_world = camera.getPos(render)
    heading = np.asarray(vehicle.heading, dtype=float)
    toward_camera = np.asarray([camera_world.x - centre.x, camera_world.y - centre.y], dtype=float)
    direction = 1.0 if float(np.dot(toward_camera, heading)) >= 0.0 else -1.0
    half_length = float(vehicle.LENGTH) / 2.0
    surface = Point3(centre.x + direction * heading[0] * half_length,
                     centre.y + direction * heading[1] * half_length,
                     centre.z)
    return surface, half_length


def is_occluded_by_nearer_vehicle(candidate: dict, candidates: list[dict], focal_px: float) -> bool:
    """Conservatively reject a target covered by a nearer vehicle on screen."""
    for nearer in candidates:
        if nearer is candidate or nearer["forward_m"] >= candidate["forward_m"] - 0.5:
            continue
        half_width = focal_px * (float(nearer["vehicle"].WIDTH) / 2.0) / nearer["forward_m"]
        height = float(getattr(nearer["vehicle"], "HEIGHT", 1.5))
        half_height = focal_px * (height / 2.0) / nearer["forward_m"]
        if abs(nearer["u"] - candidate["u"]) <= half_width and abs(nearer["v"] - candidate["v"]) <= half_height:
            return True
    return False


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if not 0.0 <= args.traffic_density <= 1.0:
        raise ValueError("--traffic-density must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    env = MetaDriveEnv({
        "manual_control": False, "use_render": False, "image_observation": True,
        "norm_pixel": False, "traffic_density": args.traffic_density, "traffic_mode": "respawn",
        "map": 3, "start_seed": args.start_seed, "num_scenarios": 1,
        "horizon": max(args.steps + 10, 100),
        "crash_vehicle_done": False, "out_of_road_done": False,
        "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False,
                           "show_line_to_navi_mark": False, "show_navigation_arrow": False},
        "sensors": {name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    })
    rows: list[dict[str, str]] = []
    try:
        env.reset()
        mount_camera_rig(env)
        matcher = build_stereo_matcher()
        camera = env.engine.get_sensor("front_left_camera").get_cam()
        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            depth = front_stereo_depth(
                frame_from_sensor(env, "front_left_camera"),
                frame_from_sensor(env, "front_right_camera"), matcher,
            )
            candidates = []
            for vehicle in getattr(env.engine.traffic_manager, "_traffic_vehicles", []):
                surface, half_length = visible_surface_point(vehicle, camera, env.engine.render)
                projected = project_world_point(camera, env.engine.render, surface)
                if projected is None:
                    continue
                u, v, truth_m = projected
                if not 1.0 <= truth_m <= 30.0:
                    continue
                candidates.append({"vehicle": vehicle, "half_length": half_length, "u": u, "v": v, "forward_m": truth_m})
            for candidate in candidates:
                if is_occluded_by_nearer_vehicle(candidate, candidates, FOCAL_LENGTH_PX):
                    continue
                vehicle, half_length = candidate["vehicle"], candidate["half_length"]
                u, v, truth_m = candidate["u"], candidate["v"], candidate["forward_m"]
                estimate_m = sample_depth(depth, u, v)
                if estimate_m is None:
                    continue
                rows.append({
                    "step": str(step), "object_id": str(vehicle.id), "object_type": type(vehicle).__name__,
                    "pixel_u": f"{u:.2f}", "pixel_v": f"{v:.2f}",
                    "metadrive_camera_forward_surface_m": f"{truth_m:.3f}",
                    "vehicle_half_length_m": f"{half_length:.3f}",
                    "stereo_sgbm_depth_m": f"{estimate_m:.3f}",
                    "absolute_error_m": f"{abs(estimate_m - truth_m):.3f}",
                })
            if terminated or truncated:
                env.reset()
                mount_camera_rig(env)
    finally:
        env.close()

    csv_path = args.output_dir / "stereo_vs_metadrive_pose_reference.csv"
    fieldnames = ["step", "object_id", "object_type", "pixel_u", "pixel_v", "metadrive_camera_forward_surface_m", "vehicle_half_length_m", "stereo_sgbm_depth_m", "absolute_error_m"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if not rows:
        print(f"No valid visible SGBM/traffic comparisons were produced. CSV: {csv_path}")
        return 2
    errors = [float(row["absolute_error_m"]) for row in rows]
    print(f"MetaDrive pose-reference comparisons: {len(rows)}")
    print(f"MAE: {sum(errors) / len(errors):.3f} m | median AE: {median(errors):.3f} m | max AE: {max(errors):.3f} m")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
