"""
Manual MetaDrive control with BEV visualization, keyboard input, weather augmentation, and YOLO detection.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import json
from typing import Tuple

def _nms_boxes(detections: List[Dict[str, Any]], iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
    if not detections:
        return []
    boxes = []
    scores = []
    for d in detections:
        boxes.append([float(d.get("xmin", 0)), float(d.get("ymin", 0)), float(d.get("xmax", 0)), float(d.get("ymax", 0))])
        scores.append(float(d.get("confidence", 0.0)))
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_others - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]

from weather.weather_utils import apply_weather, prepare_image
from yolo.yolo_utils import TRAFFIC_CLASS_NAMES, ensure_yolo_model, run_yolo, annotate_image
from yolo_depth_realtime_logger import RealtimeYOLODepthLogger

try:
    import keyboard
except ImportError:
    print("Installing keyboard package...")
    import subprocess
    import sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "keyboard"])
    import keyboard



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual MetaDrive drive with weather and YOLO detection.")
    parser.add_argument("--traffic-density", type=float, default=0.2,
                        help="Traffic density for MetaDrive environment.")
    parser.add_argument("--weather", choices=["none", "fog", "rain", "all"], default="none",
                        help="Weather augmentation to apply to the manual camera view.")
    parser.add_argument("--level", type=float, default=0.5,
                        help="Weather intensity level between 0 and 1.")
    parser.add_argument("--width", type=int, default=480,
                        help="Output frame width for display and saving.")
    parser.add_argument("--height", type=int, default=270,
                        help="Output frame height for display and saving.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for YOLO inference.")
    parser.add_argument("--confidence-threshold", type=float, default=0.10,
                        help="Minimum detection confidence for YOLO boxes.")
    parser.add_argument("--inference-size", type=int, default=960,
                        help="YOLO inference resolution. Higher values improve small/distant vehicle recall.")
    parser.add_argument("--yolo-model", default="yolov8n.pt",
                        help="Pretrained Ultralytics YOLO weights to use (default: yolov8n.pt).")
    parser.add_argument("--yolo-weights", type=Path, default=None,
                        help="Path to local pretrained .pt weights; overrides --yolo-model.")
    parser.add_argument("--save-annotated", action="store_true",
                        help="Save annotated frames to the output directory.")
    parser.add_argument("--debug-detections", action="store_true",
                        help="Print raw detection dicts periodically for debugging.")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable heuristic filtering of likely false-positive detections.")
    parser.add_argument("--output-dir", type=Path, default=Path("manual_drive_output"),
                        help="Directory to save annotated frames and detection metrics.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Maximum number of manual drive steps to run (0 = unlimited).")
    return parser.parse_args()


def get_keyboard_action(state=None) -> np.ndarray:
    steer = 0.0
    throttle = 0.0
    if keyboard.is_pressed('w'):
        throttle = 0.5
    if keyboard.is_pressed('s'):
        throttle = -0.3
    if keyboard.is_pressed('a'):
        steer = -0.5
    if keyboard.is_pressed('d'):
        steer = 0.5
    return np.array([steer, throttle], dtype=np.float32)


def extract_state_from_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        state = obs.get("state", None)
        if state is not None:
            return np.asarray(state, dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def is_traffic_detection(label: str) -> bool:
    return label in TRAFFIC_CLASS_NAMES


def is_ego_car_detection(detection: Dict[str, Any], image_shape: Tuple[int, int, int]) -> bool:
    label = detection["label"]
    if label not in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return False
    h, w = image_shape[:2]
    xmin, ymin, xmax, ymax = detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]
    box_w = xmax - xmin
    box_h = ymax - ymin
    box_area = box_w * box_h
    frame_area = float(w * h)
    center_x = (xmin + xmax) / 2.0
    box_center_y = (ymin + ymax) / 2.0
    bottom_centered = abs(center_x - 0.5 * w) < 0.15 * w
    bottom_overlap = ymax > 0.85 * h
    tall_box = box_h > 0.45 * h
    large_box = box_area > 0.15 * frame_area

    # Only suppress if the box is very large and centered on the bottom region.
    if bottom_centered and bottom_overlap and large_box:
        return True
    if bottom_centered and tall_box and box_area > 0.12 * frame_area:
        return True
    return False


def display_text(image: np.ndarray, lines: List[str]) -> np.ndarray:
    out = image.copy()
    y = 20
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22
    return out


def save_results(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "label", "confidence", "xmin", "ymin", "xmax", "ymax", "weather", "level"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Load PyTorch/YOLO before Panda3D (which MetaDrive imports).  On Windows,
    # Panda3D-first initialization can prevent PyTorch's c10.dll from loading.
    yolo_model = ensure_yolo_model(args.device, args.yolo_model, args.yolo_weights, args.confidence_threshold)
    yolo_model.inference_size = int(args.inference_size)

    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    config = {
        "manual_control": False,
        "use_render": False,
        "window_size": (args.width, args.height),
        "traffic_density": float(args.traffic_density),
        "image_observation": True,
        "norm_pixel": False,
        "num_scenarios": 1,
        "horizon": 99999,
        "crash_vehicle_done": False,
        "out_of_road_done": False,
        "truncate_as_terminate": False,
        "vehicle_config": {
            "show_navi_mark": False,
            "show_line_to_navi_mark": False,
            "show_navigation_arrow": False,
            "image_source": "rgb_camera",
        },
        "sensors": {
            "rgb_camera": (RGBCamera, args.width, args.height),
        },
    }

    env = MetaDriveEnv(config)
    obs, info = env.reset()
    print("=" * 60)
    print("MetaDrive Manual Drive + Weather + YOLO")
    print("=" * 60)
    print("Controls:")
    print("  W/S - Throttle forward/backward")
    print("  A/D - Steer left/right")
    print("  Q   - Quit")
    print("=" * 60)

    window_name = "Manual Drive YOLO"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width * 2, args.height * 2)

    done = False
    simulation_steps = 0
    control_actions = 0
    distance_m = 0.0
    previous_position = None
    controls_were_active = False
    reward = 0.0
    rows: List[Dict[str, Any]] = []
    state = extract_state_from_obs(obs)
    
    # Initialize YOLO + Depth logger
    depth_logger = RealtimeYOLODepthLogger(
        model_path=str(args.yolo_weights or args.yolo_model),
        conf_threshold=args.confidence_threshold,
        output_dir=str(args.output_dir / 'depth_logs')
    )
    print("\n✓ YOLO + Depth logger initialized")

    try:
        while not done and (args.max_steps == 0 or simulation_steps < args.max_steps):
            action = get_keyboard_action(state)
            controls_active = bool(np.any(np.abs(action) > 1e-4))
            # A control action is a new user input, not an iteration of the
            # render/physics loop. Holding W therefore counts once.
            if controls_active and not controls_were_active:
                control_actions += 1
            controls_were_active = controls_active
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            state = extract_state_from_obs(obs)

            if isinstance(obs, dict) and "image" in obs:
                frame = prepare_image(obs["image"])
            else:
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                cv2.putText(frame, "No image obs", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            clean_frame = cv2.resize(frame, (args.width, args.height))
            # Rain/fog is a visual simulation overlay. Feeding it to YOLO made
            # real confidence collapse on MetaDrive's already synthetic scene.
            # Detection uses the unmodified sensor pixels; the display still
            # shows exactly the weather condition selected by the user.
            detections = run_yolo(yolo_model, clean_frame)
            frame = apply_weather(clean_frame, args.weather, args.level)
            
            # Log depths for all detections
            depth_detections = depth_logger.process_frame(clean_frame, simulation_steps)

            if args.debug_detections and (simulation_steps % 10 == 0):
                try:
                    print(f"Simulation step {simulation_steps}: raw detections (count={len(detections)}):")
                    for i, d in enumerate(detections[:12]):
                        xmin = d.get("xmin")
                        ymin = d.get("ymin")
                        xmax = d.get("xmax")
                        ymax = d.get("ymax")
                        conf = d.get("confidence")
                        label = d.get("label")
                        print(f"  [{i}] {label} conf={conf:.2f} xmin={xmin} ymin={ymin} xmax={xmax} ymax={ymax}")
                except Exception:
                    print("Debug print failed for detections")
                # Also write raw detections JSON for later inspection
                try:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    raw_path = args.output_dir / f"raw_detections_step_{simulation_steps:04d}.json"
                    meta = {"simulation_step": simulation_steps, "model": getattr(yolo_model, "model_name", "unknown"), "detections": detections}
                    with raw_path.open("w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=2)
                except Exception:
                    pass

            traffic_candidates = [d for d in detections if is_traffic_detection(d["label"])]
            traffic_detections = [
                d for d in traffic_candidates
                if not is_ego_car_detection(d, frame.shape)
            ]

            # Apply NMS then heuristic filtering to suppress detections likely on sky/horizon/mountains
            filter_enabled = not getattr(args, "no_filter", False)
            if traffic_detections:
                # first remove duplicate/overlapping boxes
                traffic_detections = _nms_boxes(traffic_detections, iou_threshold=0.45)

            if filter_enabled and traffic_detections:
                h, w = frame.shape[:2]
                def is_reasonable_detection(d: Dict[str, Any]) -> bool:
                    try:
                        xmin = float(d.get("xmin", 0))
                        ymin = float(d.get("ymin", 0))
                        xmax = float(d.get("xmax", 0))
                        ymax = float(d.get("ymax", 0))
                    except Exception:
                        return False
                    if xmax <= xmin or ymax <= ymin:
                        return False
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    area = box_w * box_h
                    frame_area = float(w * h)
                    area_ratio = area / frame_area if frame_area > 0 else 0.0
                    # Reject extremely small or extremely large boxes
                    if area_ratio < 0.0003 or area_ratio > 0.25:
                        return False
                    # Reject boxes whose center is above the horizon (top portion) unless very confident
                    center_y = (ymin + ymax) / 2.0
                    if center_y < 0.40 * h and float(d.get("confidence", 0.0)) < 0.60:
                        return False
                    # Reasonable aspect ratio for vehicle bounding boxes
                    aspect = box_w / max(box_h, 1.0)
                    if aspect < 0.2 or aspect > 4.0:
                        return False
                    return True

                if args.debug_detections:
                    kept = []
                    for d in traffic_detections:
                        if is_reasonable_detection(d):
                            kept.append(d)
                        else:
                            print(f"Filtered detection at simulation step {simulation_steps}: {d.get('label')} conf={d.get('confidence'):.2f} bbox=({d.get('xmin')},{d.get('ymin')},{d.get('xmax')},{d.get('ymax')})")
                    traffic_detections = kept
                else:
                    traffic_detections = [d for d in traffic_detections if is_reasonable_detection(d)]
            if not traffic_detections:
                if traffic_candidates:
                    print(f"Simulation step {simulation_steps}: only ego-car traffic candidates were suppressed:")
                    for det in sorted(traffic_candidates, key=lambda x: x["confidence"], reverse=True)[:5]:
                        print(f"  {det['label']} {det['confidence']:.2f} @ {det['xmin']:.0f},{det['ymin']:.0f},{det['xmax']:.0f},{det['ymax']:.0f}")
                elif detections:
                    print(f"Simulation step {simulation_steps}: no traffic candidates; raw top model labels:")
                    for det in sorted(detections, key=lambda x: x["confidence"], reverse=True)[:5]:
                        print(f"  {det['label']} {det['confidence']:.2f}")

            position = getattr(getattr(env, "agent", None), "position", None)
            if isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 2:
                position = np.asarray(position[:2], dtype=np.float64)
                if previous_position is not None:
                    distance_m += float(np.linalg.norm(position - previous_position))
                previous_position = position

            annotated = annotate_image(frame, traffic_detections)
            speed = float(info.get("velocity", 0.0)) if isinstance(info, dict) else 0.0
            status = [
                f"Control actions: {control_actions}",
                f"Distance: {distance_m:.2f} m",
                f"Speed: {speed:.2f}",
                f"Traffic density: {args.traffic_density:.2f}",
                f"Weather: {args.weather} ({args.level:.2f})",
                f"Traffic detections: {len(traffic_detections)}",
            ]
            annotated = display_text(annotated, status)
            cv2.imshow(window_name, annotated)

            if args.save_annotated:
                image_path = args.output_dir / f"step_{simulation_steps:04d}.png"
                cv2.imwrite(str(image_path), annotated)

            for det in traffic_detections:
                rows.append({
                    "step": simulation_steps,
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "xmin": det["xmin"],
                    "ymin": det["ymin"],
                    "xmax": det["xmax"],
                    "ymax": det["ymax"],
                    "weather": args.weather,
                    "level": args.level,
                })

            if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
                print("Quit requested by user.")
                break

            simulation_steps += 1

            if done:
                reason = []
                if terminated:
                    reason.append("terminated")
                if truncated:
                    reason.append("truncated")
                if info:
                    for key in ["crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road", "arrive_dest", "max_step"]:
                        if info.get(key):
                            reason.append(key)
                print("Episode ending because:", ", ".join(reason) if reason else "unknown")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        depth_logger.close()  # Save depth logs

    if rows:
        save_results(args.output_dir / "manual_drive_detections.csv", rows)
        print(f"Saved detection metrics to {args.output_dir / 'manual_drive_detections.csv'}")

    print(f"Episode ended. Control actions: {control_actions}, simulation steps: {simulation_steps}, distance: {distance_m:.2f} m, final reward: {reward:.3f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
