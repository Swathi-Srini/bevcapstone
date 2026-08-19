"""Live four-direction camera rig: front stereo depth + YOLO + weather.

The front direction comprises the physical front-left/front-right stereo pair;
the remaining streams are left, right, and rear monocular cameras.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from weather.weather_utils import apply_weather
from yolo.yolo_utils import ensure_yolo_model, run_yolo


# Technical specification: 1200x900, horizontal FOV 60 deg, B=0.5m.
CAM_W, CAM_H, CAM_FOV = 1200, 900, 60
FOCAL_LENGTH_PX = 1000.0
STEREO_BASELINE_M = 0.5
CAMERA_RIGS = {
    "front_left_camera": ((-0.25, 2.0, 1.4), (0, -5, 0)),
    "front_right_camera": ((0.25, 2.0, 1.4), (0, -5, 0)),
    "left_camera": ((0.0, 0.0, 1.4), (-90, -5, 0)),
    "right_camera": ((0.0, 0.0, 1.4), (90, -5, 0)),
    "rear_camera": ((0.0, -2.0, 1.4), (180, -5, 0)),
}
# Four logical YOLO views from the architecture: front primary, left, right,
# and rear. The front-right sensor is used only as the stereo partner, not as
# a duplicate front-object detector.
YOLO_CAMERAS = ("front_left_camera", "left_camera", "right_camera", "rear_camera")
CAMERA_HEIGHT_M = 1.4
DOWNWARD_PITCH_DEG = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual four-direction YOLO, weather, and front stereo-depth visualization.")
    parser.add_argument("--weather", choices=("none", "fog", "rain", "all"), default="none")
    parser.add_argument("--level", type=float, default=0.5, help="Weather intensity: 0 to 1.")
    parser.add_argument("--traffic-density", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--inference-size", type=int, default=1216,
                        help="YOLO input size; 1216 is divisible by the model stride of 32.")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("integrated_output"),
                        help="Folder for detection_log.csv.")
    parser.add_argument("--perception-weather", action="store_true",
                        help="Apply weather to YOLO and stereo too. This deliberately degrades perception; by default weather is display-only.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until Q is pressed.")
    return parser.parse_args()


def build_stereo_matcher() -> cv2.StereoSGBM:
    """Exact StereoSGBM values specified in Table 5."""
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=192, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def frame_from_sensor(env, name: str) -> np.ndarray:
    frame = env.engine.get_sensor(name).get_rgb_array_cpu()
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def front_stereo_depth(left_bgr: np.ndarray, right_bgr: np.ndarray, matcher: cv2.StereoSGBM) -> np.ndarray:
    """Z = f * B / d_px, with f=1000 px and B=0.5 m."""
    # The cameras are mounted parallel with identical intrinsics, so their
    # simulator images are already rectified. Do not inject random weather
    # separately into the pair before matching: unmatched rain streaks produce
    # false disparity everywhere.
    disparity_raw = matcher.compute(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY),
                                    cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY))
    # Speckle removal is post-processing, not a change to the Table-5 SGBM
    # parameters. It removes isolated invalid matches from uniform road/sky.
    cv2.filterSpeckles(disparity_raw, 0, 50, 2)
    disparity = disparity_raw.astype(np.float32) / 16.0
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0.0
    depth[valid] = (FOCAL_LENGTH_PX * STEREO_BASELINE_M) / disparity[valid]
    depth[(depth < 1.0) | (depth > 30.0)] = 0.0
    return depth


def object_stereo_depth(detection: dict, depth: np.ndarray) -> float | None:
    """Robust median of valid stereo pixels inside the detected object."""
    h, w = depth.shape
    x1 = int(np.clip(detection["xmin"], 0, w - 1))
    y1 = int(np.clip(detection["ymin"], 0, h - 1))
    x2 = int(np.clip(detection["xmax"], x1 + 1, w))
    y2 = int(np.clip(detection["ymax"], y1 + 1, h))
    # Ignore the box edges/background. The central vehicle region is more
    # likely to contain valid SGM correspondences than its road-contact pixel.
    margin_x = max(1, int((x2 - x1) * 0.15))
    margin_y = max(1, int((y2 - y1) * 0.12))
    patch = depth[y1 + margin_y:max(y1 + margin_y + 1, y2 - margin_y),
                  x1 + margin_x:max(x1 + margin_x + 1, x2 - margin_x)]
    values = patch[patch > 0]
    return None if values.size == 0 else float(np.median(values))


def depth_label(detection: dict, depth: np.ndarray) -> str:
    value = object_stereo_depth(detection, depth)
    return "--" if value is None else f"{value:.1f}m"


def ego_position_from_camera(camera: str, u: float, forward_m: float) -> tuple[float, float]:
    """Convert a camera pixel/depth to ego coordinates: x=right, y=forward."""
    position, hpr = CAMERA_RIGS[camera]
    yaw = math.radians(hpr[0])
    lateral_m = (u - CAM_W / 2.0) * forward_m / FOCAL_LENGTH_PX
    # Camera forward and right vectors in MetaDrive's x=right, y=forward axes.
    forward_x, forward_y = math.sin(yaw), math.cos(yaw)
    right_x, right_y = math.cos(yaw), -math.sin(yaw)
    return (position[0] + forward_x * forward_m + right_x * lateral_m,
            position[1] + forward_y * forward_m + right_y * lateral_m)


def estimate_detection_position(camera: str, detection: dict, stereo_depth: np.ndarray) -> tuple[str, float | None, float | None, float | None]:
    """Stereo for front; ground-plane monocular depth + ego position for side/rear."""
    u = (float(detection["xmin"]) + float(detection["xmax"])) / 2.0
    if camera == "front_left_camera":
        depth_m = object_stereo_depth(detection, stereo_depth)
        if depth_m is None:
            return "stereo_sgm_invalid", None, None, None
        x_ego, forward_ego = ego_position_from_camera(camera, u, depth_m)
        return "stereo_sgm", depth_m, x_ego, forward_ego

    # Ground-plane projection from the bounding-box bottom centre:
    # Z = H / tan(phi), where phi is the downward viewing angle, then
    # X=(u-cx)Z/f. This is the monocular / 3D-position stage in the diagram.
    v = float(detection["ymax"])
    downward_angle = math.radians(DOWNWARD_PITCH_DEG) + math.atan2(v - CAM_H / 2.0, FOCAL_LENGTH_PX)
    if downward_angle <= 0.0:
        return "ground_plane_invalid", None, None, None
    depth_m = CAMERA_HEIGHT_M / math.tan(downward_angle)
    if not 1.0 <= depth_m <= 80.0:
        return "ground_plane_invalid", None, None, None
    x_ego, forward_ego = ego_position_from_camera(camera, u, depth_m)
    return "ground_plane_monocular", depth_m, x_ego, forward_ego


def annotate_with_depth(image: np.ndarray, detections: List[dict], depth: np.ndarray | None) -> np.ndarray:
    """Draw compact YOLO labels. Stereo depth remains in the terminal/CSV log."""
    out = image.copy()
    for det in detections:
        x1 = int(np.clip(det["xmin"], 0, out.shape[1] - 1))
        y1 = int(np.clip(det["ymin"], 0, out.shape[0] - 1))
        x2 = int(np.clip(det["xmax"], 0, out.shape[1] - 1))
        y2 = int(np.clip(det["ymax"], 0, out.shape[0] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        text = f"{det['label']} | conf {float(det['confidence']):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)
        label_y = max(text_h + baseline + 4, y1 - 4)
        cv2.rectangle(out, (x1, label_y - text_h - baseline - 4), (min(out.shape[1] - 1, x1 + text_w + 6), label_y), (0, 0, 0), -1)
        cv2.putText(out, text, (x1 + 3, label_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def tile(image: np.ndarray, title: str, width: int = 400, height: int = 300) -> np.ndarray:
    out = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (width, 27), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def depth_visualization(depth: np.ndarray) -> np.ndarray:
    scaled = np.clip(depth / 30.0 * 255.0, 0, 255).astype(np.uint8)
    out = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
    out[depth <= 0] = 0
    return out


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.level <= 1.0:
        raise ValueError("--level must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = ensure_yolo_model(args.device, args.yolo_model, None, args.confidence_threshold)
    model.inference_size = args.inference_size

    # Import after torch/YOLO: avoids a Windows Panda3D/PyTorch DLL conflict.
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    env = MetaDriveEnv({
        "manual_control": False, "use_render": False, "image_observation": True,
        "norm_pixel": False, "traffic_density": args.traffic_density, "num_scenarios": 1,
        "horizon": 99999, "crash_vehicle_done": False, "out_of_road_done": False,
        "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False,
                           "show_line_to_navi_mark": False, "show_navigation_arrow": False},
        "sensors": {name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    })
    env.reset()
    for name, (position, hpr) in CAMERA_RIGS.items():
        env.engine.get_sensor(name).track(env.agent.origin, position, hpr)

    matcher = build_stereo_matcher()
    window = "Four-camera YOLO + Weather + Front Stereo Depth (WASD, Q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1200, 600)
    print("Controls in this visualization window: W/S throttle, A/D steer, Q quit.")
    print("Front-left + front-right -> StereoSGBM -> Z=1000*0.5/disparity. YOLO runs on four logical views: front, left, right, rear.")
    print("Weather is display-only by default; use --perception-weather to intentionally test weather-degraded YOLO/stereo.")
    step = 0
    log_rows: List[dict] = []
    try:
        while args.max_steps == 0 or step < args.max_steps:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            action = np.array([(-0.5 if key == ord("a") else 0.5 if key == ord("d") else 0.0),
                               (0.5 if key == ord("w") else -0.3 if key == ord("s") else 0.0)], dtype=np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            frames: Dict[str, np.ndarray] = {name: frame_from_sensor(env, name) for name in CAMERA_RIGS}
            weathered = {name: apply_weather(frame, args.weather, args.level) for name, frame in frames.items()}
            perception = weathered if args.perception_weather else frames
            depth = front_stereo_depth(perception["front_left_camera"], perception["front_right_camera"], matcher)
            detections = {name: (run_yolo(model, perception[name]) if name in YOLO_CAMERAS else [])
                          for name in CAMERA_RIGS}
            front = annotate_with_depth(weathered["front_left_camera"], detections["front_left_camera"], depth)
            grid = np.vstack((np.hstack((tile(front, "FRONT LEFT: YOLO + stereo Z"),
                                        tile(weathered["front_right_camera"], "FRONT RIGHT: stereo partner (no YOLO)"),
                                        tile(depth_visualization(depth), "FRONT STEREO DEPTH: 1-30 m"))),
                              np.hstack((tile(annotate_with_depth(weathered["left_camera"], detections["left_camera"], None), "LEFT: YOLO monocular"),
                                        tile(annotate_with_depth(weathered["right_camera"], detections["right_camera"], None), "RIGHT: YOLO monocular"),
                                        tile(annotate_with_depth(weathered["rear_camera"], detections["rear_camera"], None), "REAR: YOLO monocular")))))
            cv2.imshow(window, grid)
            terminal_rows = []
            for camera, camera_detections in detections.items():
                for det in camera_detections:
                    depth_method, depth_m, ego_x_m, ego_forward_m = estimate_detection_position(camera, det, depth)
                    depth_text = f"{depth_m:.1f}" if depth_m is not None else "--"
                    x_text = f"{ego_x_m:+.1f}" if ego_x_m is not None else "--"
                    forward_text = f"{ego_forward_m:+.1f}" if ego_forward_m is not None else "--"
                    terminal_rows.append((camera, det["label"], float(det["confidence"]), depth_method, depth_text, x_text, forward_text))
                    log_rows.append({
                        "timestamp_unix": f"{time.time():.3f}", "step": step, "camera": camera,
                        "object": det["label"], "confidence": f"{float(det['confidence']):.4f}",
                        "depth_method": depth_method, "depth_m": depth_text,
                        "ego_x_right_m": x_text, "ego_forward_m": forward_text,
                        "xmin": f"{det['xmin']:.1f}", "ymin": f"{det['ymin']:.1f}",
                        "xmax": f"{det['xmax']:.1f}", "ymax": f"{det['ymax']:.1f}",
                        "weather_display": args.weather, "weather_level": args.level,
                        "perception_source": "weathered" if args.perception_weather else "clean",
                    })
            if terminal_rows:
                print(f"\nSTEP {step}  WEATHER={args.weather}({args.level:.2f})  PERCEPTION={'weathered' if args.perception_weather else 'clean'}")
                print(f"{'CAMERA':<20} {'OBJECT':<9} {'CONF':>6} {'DEPTH METHOD':<24} {'DEPTH':>7} {'X RIGHT':>8} {'FORWARD':>8}")
                print("-" * 96)
                for camera, obj, confidence, method, depth_value, x_value, forward_value in terminal_rows:
                    print(f"{camera:<20} {obj:<9} {confidence:>6.2f} {method:<24} {depth_value:>6}m {x_value:>7}m {forward_value:>7}m")
            elif step % 30 == 0:
                print(f"[step {step}] no YOLO traffic detections", flush=True)
            step += 1
            if terminated or truncated:
                env.reset()
                for name, (position, hpr) in CAMERA_RIGS.items():
                    env.engine.get_sensor(name).track(env.agent.origin, position, hpr)
    finally:
        env.close()
        cv2.destroyAllWindows()
        if log_rows:
            log_path = args.output_dir / "detection_log.csv"
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            print(f"Saved {len(log_rows)} detection rows to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
