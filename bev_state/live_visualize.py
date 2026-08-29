"""Live MetaDrive perception-to-BEV validation visualiser.

This is a system-test tool, not a training collector.  It runs the existing
camera, YOLO/stereo, and BEV-state modules in one process so a reviewer can
inspect whether detected objects land plausibly in the policy input.
"""

from __future__ import annotations

import argparse
import math

# On Windows, initialise Torch before OpenCV/Panda3D-facing modules.  The live
# perception runner follows the same ordering to avoid a c10.dll initialisation
# failure caused by native DLL load order.
import torch  # noqa: F401
import cv2
import numpy as np

from bev_state import BEVStateAssembler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", choices=("none", "fog", "rain"), default="none")
    parser.add_argument("--level", type=float, default=0.5, help="Synthetic weather intensity in [0,1].")
    parser.add_argument("--perception-weather", action="store_true",
                        help="Feed weathered images to YOLO and stereo. Without it weather is display-only.")
    parser.add_argument("--traffic-density", type=float, default=0.35)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until Q is pressed.")
    parser.add_argument("--spawn-target-distance", type=float, default=0.0,
                        help="System-test only: spawn a stationary vehicle this many metres ahead in the ego lane.")
    parser.add_argument("--auto-drive", action="store_true",
                        help="Use MetaDrive IDM for the ego vehicle; useful with --spawn-target-distance.")
    return parser.parse_args()


def colourise_bev(grid: np.ndarray) -> np.ndarray:
    """Render each 64x64 BEV cell as an exact 8x8 display square.

    ``INTER_NEAREST`` is deliberate: this is a visual magnification only.  No
    interpolation, crop, or geometric rescaling is applied to the policy grid.
    """

    image = np.zeros((*grid.shape, 3), dtype=np.uint8)
    image[np.isclose(grid, -1.0)] = (58, 58, 58)       # unknown: charcoal
    image[np.isclose(grid, 0.0)] = (20, 20, 35)        # visible free: navy-black
    image[np.isclose(grid, 0.5)] = (0, 220, 255)       # route/lane: yellow (BGR)
    image[np.isclose(grid, 0.8)] = (0, 145, 255)       # boundary: orange (when supplied)
    image[np.isclose(grid, 0.9)] = (70, 235, 70)       # ego: green
    image[np.isclose(grid, 1.0)] = (45, 45, 235)       # occupied object: red
    cell_pixels = 8
    image = cv2.resize(image, (64 * cell_pixels, 64 * cell_pixels), interpolation=cv2.INTER_NEAREST)
    # Fine lines make individual cells inspectable. Thick lines occur every
    # eight cells (2.5m), because one cell is 0.3125m in the current contract.
    for index in range(65):
        coordinate = min(index * cell_pixels, image.shape[0] - 1)
        colour = (70, 70, 70) if index % 8 == 0 else (35, 35, 35)
        thickness = 1 if index % 8 else 2
        cv2.line(image, (coordinate, 0), (coordinate, image.shape[0] - 1), colour, thickness)
        cv2.line(image, (0, coordinate), (image.shape[1] - 1, coordinate), colour, thickness)
    cv2.putText(image, "far +17.5m", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "ego row 56", (8, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return image


def add_text_panel(bev_image: np.ndarray, state, weather: str, perception_weather: bool) -> np.ndarray:
    panel = np.full((512, 480, 3), 20, dtype=np.uint8)
    lines = [
        "BEV state integration test",
        f"weather={weather}; perception={'weathered' if perception_weather else 'clean'}",
        "grid: unknown charcoal | free navy | ego green | occupied red",
        "route/lane yellow | boundary orange (when those BEV layers exist)",
        "display: 64x64 cells, enlarged 8x with nearest-neighbour only",
        "extent: x=-10..+10m; y=-2.5..+17.5m; 0.3125m/cell",
        "",
        "scalar state:",
        f" speed_mps                 {state.scalar_state[0]:8.3f}",
        f" route_progress            {state.scalar_state[1]:8.3f}",
        f" lateral_deviation_m       {state.scalar_state[2]:8.3f}",
        f" heading_error_rad         {state.scalar_state[3]:8.3f}",
        f" curvature_ahead_rad_per_m {state.scalar_state[4]:8.4f}",
        f" distance_to_goal_m        {state.scalar_state[5]:8.3f}",
        "",
        f"projected objects: {len(state.objects)}",
    ]
    for obj in state.objects[:10]:
        lines.append(f" {obj.label:<9} x={obj.x_right_m:+5.1f} y={obj.y_forward_m:+5.1f} {obj.depth_method}")
    if len(state.objects) > 10:
        lines.append(f" ... {len(state.objects) - 10} additional objects")
    lines += ["", "W/A/S/D drive; Q quit", "Object outside 17.5m front grid is not painted."]
    for row, line in enumerate(lines):
        y = 24 + row * 20
        if y >= 505:
            break
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA)
    return np.hstack((bev_image, panel))


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.level <= 1.0:
        raise ValueError("--level must be between 0 and 1.")
    if args.spawn_target_distance and not 3.0 <= args.spawn_target_distance <= 15.0:
        raise ValueError("--spawn-target-distance must be between 3m and 15m so it remains inside the BEV.")

    # These imports are intentionally delayed: on some Windows configurations
    # importing Torch before Panda3D avoids or mitigates DLL initialisation issues.
    from manual_drive_stereo_yolo_weather import (
        CAMERA_RIGS,
        YOLO_CAMERAS,
        annotate_with_depth,
        apply_synchronized_stereo_weather,
        frame_from_sensor,
        mount_camera_rig,
    )
    from weather.weather_utils import apply_weather
    from yolo.yolo_utils import ensure_yolo_model, run_yolo
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.component.vehicle.vehicle_type import SVehicle
    from metadrive.policy.idm_policy import IDMPolicy
    from manual_drive_stereo_yolo_weather import CAM_H, CAM_W

    model = ensure_yolo_model(args.device, args.yolo_model, None, 0.4)
    config = {
        "manual_control": False, "use_render": False, "image_observation": True,
        "norm_pixel": False, "traffic_density": args.traffic_density, "num_scenarios": 1,
        "horizon": 99999, "crash_vehicle_done": False, "out_of_road_done": False,
        "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False,
                           "show_line_to_navi_mark": False, "show_navigation_arrow": False},
        "sensors": {name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    }
    if args.auto_drive:
        config["agent_policy"] = IDMPolicy
    env = MetaDriveEnv(config)
    _, info = env.reset()
    mount_camera_rig(env)

    def spawn_controlled_target() -> None:
        """Place a static validation target in front of ego without using it as policy input."""
        if not args.spawn_target_distance:
            return
        lane = env.agent.lane
        longitudinal, _ = lane.local_coordinates(env.agent.position)
        env.engine.spawn_object(
            SVehicle,
            vehicle_config={"spawn_lane_index": lane.index,
                            "spawn_longitude": float(longitudinal + args.spawn_target_distance)},
            random_seed=17,
        )
        print(f"Spawned stationary BEV validation vehicle {args.spawn_target_distance:.1f}m ahead.")

    spawn_controlled_target()
    assembler = BEVStateAssembler()
    window = "Perception to BEV validation (WASD, Q quit)"
    # AUTOSIZE preserves the composite's native pixels.  In particular, the
    # 512x512 magnified BEV remains square instead of being stretched by an
    # arbitrary window width/height pair.
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print("Validation goal: approach a vehicle within 5-15m and check that its red BEV footprint")
    print("appears in a plausible position. Grey is unknown; black is camera-visible free space.")

    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            action = np.asarray([0.0, 0.0], dtype=np.float32) if args.auto_drive else np.asarray([
                -0.5 if key == ord("a") else 0.5 if key == ord("d") else 0.0,
                0.5 if key == ord("w") else -0.3 if key == ord("s") else 0.0,
            ], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            raw_frames = {name: frame_from_sensor(env, name) for name in CAMERA_RIGS}
            shown_frames = {name: apply_weather(frame, args.weather, args.level) for name, frame in raw_frames.items()}
            shown_frames["front_left_camera"], shown_frames["front_right_camera"] = apply_synchronized_stereo_weather(
                raw_frames["front_left_camera"], raw_frames["front_right_camera"], args.weather, args.level
            )
            perception_frames = shown_frames if args.perception_weather else raw_frames
            detections = {name: run_yolo(model, perception_frames[name]) for name in YOLO_CAMERAS}
            state = assembler.assemble(
                env=env, info=info, detections_by_camera=detections, frames=perception_frames
            )
            front = annotate_with_depth(shown_frames["front_left_camera"], detections["front_left_camera"], state.front_depth)
            front = cv2.resize(front, (683, 512), interpolation=cv2.INTER_AREA)
            dashboard = add_text_panel(colourise_bev(state.bev_grid), state, args.weather, args.perception_weather)
            top_row = np.hstack((front, dashboard))
            cv2.imshow(window, np.vstack((top_row, np.full((25, top_row.shape[1], 3), 20, dtype=np.uint8))))
            step += 1
            if terminated or truncated:
                _, info = env.reset()
                mount_camera_rig(env)
                spawn_controlled_target()
    finally:
        env.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
