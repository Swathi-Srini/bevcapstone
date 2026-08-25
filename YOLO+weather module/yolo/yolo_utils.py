"""YOLO helper functions for detection and annotation."""

from __future__ import annotations

import importlib.util
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

TRAFFIC_CLASS_NAMES = {
    # Technical specification Sec. 4: COCO IDs 0, 2, 3, 5, 7 only.
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def _import_torch() -> Any | None:
    try:
        import torch
    except Exception as exc:
        print(f"Warning: PyTorch import failed: {exc}")
        return None
    return torch


class FallbackDetector:
    """A lightweight detector that uses MetaDrive vehicle positions as boxes."""

    def __init__(self, confidence_threshold: float, device: str = "cpu"):
        self.model_name = "fallback_detector"
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.fallback_env: Optional[Any] = None
        self.current_weather: str = "none"
        self.current_level: float = 0.0

    def _get_vehicle_dimensions(self, vehicle: Any) -> tuple[float, float, float]:
        def scalar(value: Any, default: float) -> float:
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if value is None:
                return default
            try:
                return float(value)
            except Exception:
                return default

        width = getattr(vehicle, "WIDTH", None) or getattr(vehicle, "width", None)
        length = getattr(vehicle, "LENGTH", None) or getattr(vehicle, "length", None)
        height = getattr(vehicle, "HEIGHT", None) or getattr(vehicle, "height", None)
        return (
            scalar(width, 1.85),
            scalar(length, 4.5),
            scalar(height, 1.5),
        )

    def _get_vehicle_pose(self, vehicle: Any, world_np: Any):
        from panda3d.core import Point3, Quat, Vec3

        origin = getattr(vehicle, "origin", None)
        if origin is not None:
            return origin.getPos(world_np), origin.getQuat(world_np)

        pos = getattr(vehicle, "position", None)
        if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None, None

        z = getattr(vehicle, "HEIGHT", None)
        if z is None:
            z = getattr(vehicle, "height", None)
        z = float(z) / 2.0 if z is not None else 0.75
        world_pos = Point3(float(pos[0]), float(pos[1]), z)

        heading = getattr(vehicle, "heading", None)
        quat = Quat()
        if isinstance(heading, (list, tuple)) and len(heading) == 2:
            angle = math.degrees(math.atan2(float(heading[1]), float(heading[0])))
            quat.setHpr(Vec3(angle, 0.0, 0.0))
        elif isinstance(heading, (int, float)):
            angle = float(heading)
            if abs(angle) <= 2.0 * math.pi:
                angle = math.degrees(angle)
            quat.setHpr(Vec3(angle, 0.0, 0.0))
        else:
            quat.setHpr(Vec3(0.0, 0.0, 0.0))

        return world_pos, quat

    def _project_vehicle_to_image(self, vehicle: Any, image: np.ndarray) -> Optional[Dict[str, Any]]:
        if self.fallback_env is None:
            return None

        try:
            ego = getattr(self.fallback_env, "agent", None)
            if ego is None or vehicle is ego:
                return None

            engine = getattr(self.fallback_env, "engine", None)
            camera_sensor = None
            if engine is not None:
                sensors = getattr(engine, "sensors", None)
                if isinstance(sensors, dict):
                    camera_sensor = sensors.get("rgb_camera")
                if camera_sensor is None:
                    camera_sensor = getattr(engine, "rgb_camera", None)

            if camera_sensor is not None:
                camera_node = camera_sensor.get_cam()
                lens = camera_sensor.get_lens()
                world_np = getattr(engine, "worldNP", None)
                if camera_node is not None and lens is not None and world_np is not None:
                    det = self._project_vehicle_to_image_with_camera(vehicle, image, camera_node, lens, world_np)
                    if det is not None:
                        return det

            return self._project_vehicle_to_image_planar(vehicle, image, ego)
        except Exception:
            return None

    def _project_vehicle_to_image_with_camera(
        self,
        vehicle: Any,
        image: np.ndarray,
        camera_node: Any,
        lens: Any,
        world_np: Any,
    ) -> Optional[Dict[str, Any]]:
        from panda3d.core import Point2, Point3, Vec3

        world_pos, world_quat = self._get_vehicle_pose(vehicle, world_np)
        if world_pos is None or world_quat is None:
            return None

        width, length, height = self._get_vehicle_dimensions(vehicle)
        half_w, half_l, half_h = width / 2.0, length / 2.0, height / 2.0

        local_corners = [
            Vec3(-half_l, -half_w, -half_h),
            Vec3(-half_l, half_w, -half_h),
            Vec3(half_l, half_w, -half_h),
            Vec3(half_l, -half_w, -half_h),
            Vec3(-half_l, -half_w, half_h),
            Vec3(-half_l, half_w, half_h),
            Vec3(half_l, half_w, half_h),
            Vec3(half_l, -half_w, half_h),
        ]

        h, w = image.shape[:2]
        projected_pixels_inframe: List[tuple[int, int]] = []
        projected_pixels_any: List[tuple[int, int]] = []
        projected_pixels_outside: List[tuple[int, int]] = []

        for local_point in local_corners:
            world_corner = world_pos + world_quat.xform(local_point)
            cam_corner = camera_node.getRelativePoint(world_np, world_corner)
            if cam_corner.getY() <= 0:
                continue
            projected = Point2()
            if not lens.project(Point3(cam_corner), projected):
                continue
            raw_x = (projected.getX() * 0.5 + 0.5) * w
            raw_y = (0.5 - projected.getY() * 0.5) * h
            clamped_x = int(np.clip(raw_x, 0, w - 1))
            clamped_y = int(np.clip(raw_y, 0, h - 1))
            projected_pixels_any.append((clamped_x, clamped_y))
            if 0.0 <= raw_x < w and 0.0 <= raw_y < h:
                projected_pixels_inframe.append((clamped_x, clamped_y))
            else:
                projected_pixels_outside.append((clamped_x, clamped_y))

        if len(projected_pixels_inframe) < 6:
            return None
        if len(projected_pixels_any) < 6:
            return None
        if len(projected_pixels_outside) > 2:
            return None

        xs, ys = zip(*projected_pixels_inframe)
        box_x1 = min(xs)
        box_y1 = min(ys)
        box_x2 = max(xs)
        box_y2 = max(ys)
        if box_x2 - box_x1 < 16 or box_y2 - box_y1 < 16:
            return None
        if box_x1 >= w or box_y1 >= h or box_x2 <= 0 or box_y2 <= 0:
            return None

        # only keep boxes based on corners that were actually inside the frame
        if box_x1 < 0 or box_y1 < 0 or box_x2 >= w or box_y2 >= h:
            return None

        camera_center = camera_node.getRelativePoint(world_np, world_pos)
        if camera_center.getY() <= 0:
            return None

        center_proj = Point2()
        if not lens.project(Point3(camera_center), center_proj):
            return None

        depth = float(max(camera_center.getY(), 0.1))
        if depth > 100.0:
            return None

        area = float((box_x2 - box_x1) * (box_y2 - box_y1))
        area_ratio = area / float(w * h)
        if area_ratio < 0.0004:
            return None

        size_score = float(np.clip((area_ratio - 0.005) / 0.045, 0.0, 1.0))
        distance_score = float(np.clip(1.0 - (depth - 3.0) / 65.0, 0.0, 1.0))
        angle_score = float(np.clip(1.0 - abs(center_proj.getX()) / 0.9, 0.0, 1.0))
        weather_penalty = 0.0
        if self.current_weather in {"fog", "rain", "all"}:
            weather_penalty = 0.30 * self.current_level

        confidence = float(np.clip(
            0.20 + 0.40 * size_score + 0.30 * distance_score + 0.15 * angle_score - weather_penalty,
            0.05,
            0.95,
        ))
        if confidence < self.confidence_threshold:
            return None

        return {
            "xmin": float(box_x1),
            "ymin": float(box_y1),
            "xmax": float(box_x2),
            "ymax": float(box_y2),
            "confidence": confidence,
            "label": "car",
        }

    def _project_vehicle_to_image_planar(
        self,
        vehicle: Any,
        image: np.ndarray,
        ego: Any,
    ) -> Optional[Dict[str, Any]]:
        from panda3d.core import Vec3

        if ego is None:
            return None

        ego_pos = getattr(ego, "position", None)
        veh_pos = getattr(vehicle, "position", None)
        if ego_pos is None or veh_pos is None:
            return None
        if not isinstance(ego_pos, (list, tuple)) or not isinstance(veh_pos, (list, tuple)):
            return None

        heading = getattr(ego, "heading", None)
        if isinstance(heading, (list, tuple)) and len(heading) == 2:
            hx, hy = float(heading[0]), float(heading[1])
        elif isinstance(heading, (int, float)):
            angle = float(heading)
            if abs(angle) <= 2.0 * math.pi:
                angle = math.degrees(angle)
            hx = math.cos(math.radians(angle))
            hy = math.sin(math.radians(angle))
        else:
            hx, hy = 1.0, 0.0

        dx = float(veh_pos[0]) - float(ego_pos[0])
        dy = float(veh_pos[1]) - float(ego_pos[1])
        rel_x = dx * hy - dy * hx
        rel_y = dx * hx + dy * hy
        if rel_y <= 0:
            return None

        width, length, height = self._get_vehicle_dimensions(vehicle)
        h, w = image.shape[:2]
        fov_x = 60.0
        focal = (w * 0.5) / math.tan(math.radians(fov_x) / 2.0)
        pixel_width = int(np.clip(width * focal / max(rel_y, 0.1), 12, w * 0.55))
        pixel_height = int(np.clip(height * focal / max(rel_y, 0.1), 18, h * 0.6))
        center_x = int(np.clip(w * 0.5 + rel_x * focal / max(rel_y, 0.1), 0, w - 1))
        center_y = int(np.clip(h * 0.68 - pixel_height * 0.5, 0, h - 1))

        box_x1 = center_x - pixel_width // 2
        box_x2 = center_x + pixel_width // 2
        box_y2 = center_y + pixel_height // 2
        box_y1 = center_y - pixel_height // 2

        if box_x2 <= box_x1 or box_y2 <= box_y1:
            return None
        if box_x1 >= w or box_y1 >= h or box_x2 <= 0 or box_y2 <= 0:
            return None

        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(w - 1, box_x2)
        box_y2 = min(h - 1, box_y2)

        depth = float(max(rel_y, 0.1))
        if depth > 90.0:
            return None

        area_ratio = float((box_x2 - box_x1) * (box_y2 - box_y1)) / float(w * h)
        if area_ratio < 0.0005:
            return None

        size_score = float(np.clip((area_ratio - 0.005) / 0.045, 0.0, 1.0))
        distance_score = float(np.clip(1.0 - (depth - 3.0) / 65.0, 0.0, 1.0))
        angle_score = float(np.clip(1.0 - abs(rel_x) / max(rel_y, 1.0), 0.0, 1.0))
        weather_penalty = 0.0
        if self.current_weather in {"fog", "rain", "all"}:
            weather_penalty = 0.30 * self.current_level

        confidence = float(np.clip(
            0.18 + 0.40 * size_score + 0.30 * distance_score + 0.12 * angle_score - weather_penalty,
            0.05,
            0.90,
        ))
        if confidence < self.confidence_threshold:
            return None

        return {
            "xmin": float(box_x1),
            "ymin": float(box_y1),
            "xmax": float(box_x2),
            "ymax": float(box_y2),
            "confidence": confidence,
            "label": "car",
        }

    def __call__(self, image: np.ndarray):
        if self.fallback_env is None:
            return []

        detections: List[Dict[str, Any]] = []
        vehicles = []
        try:
            engine = getattr(self.fallback_env, "engine", None)
            manager = getattr(engine, "agent_manager", None)
            if manager is not None:
                vehicles = list(getattr(manager, "active_agents", {}).values())
            # Prefer explicit traffic manager lists when available (traffic vehicles spawned by the engine)
            traffic_manager = getattr(engine, "traffic_manager", None)
            if traffic_manager is not None:
                # try several common attributes used across Metadrive versions
                tm_list = getattr(traffic_manager, "vehicles", None) or getattr(traffic_manager, "traffic_vehicles", None) or getattr(traffic_manager, "_traffic_vehicles", None)
                if tm_list:
                    # merge unique vehicles
                    vehicles_by_id = {getattr(v, 'id', id(v)): v for v in vehicles}
                    for v in tm_list:
                        vehicles_by_id[getattr(v, 'id', id(v))] = v
                    vehicles = list(vehicles_by_id.values())
        except Exception:
            vehicles = []

        if not vehicles:
            return []

        ego = getattr(self.fallback_env, "agent", None)
        vehicles = [v for v in vehicles if v is not ego]

        visible_detections = []
        for vehicle in vehicles:
            det = self._project_vehicle_to_image(vehicle, image)
            if det is not None and det["confidence"] >= self.confidence_threshold:
                visible_detections.append(det)

        if not visible_detections:
            return []

        visible_detections.sort(key=lambda det: det["confidence"], reverse=True)
        if len(visible_detections) > 8:
            visible_detections = visible_detections[:8]
        return visible_detections


def ensure_yolo_model(device: str, model_name: str, weights: Path | None, confidence_threshold: float):
    """Load a pretrained Ultralytics YOLO model only.

    The manual research workflow must not substitute untrained or
    simulation-generated detections for model predictions.
    """
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError(
            "Ultralytics YOLO is required for real detections. Install it with "
            "`python -m pip install ultralytics`, then run again."
        )

    from ultralytics import YOLO
    source = str(weights) if weights is not None else model_name
    try:
        print(f"Loading pretrained YOLO model: {source}")
        model = YOLO(source)
        model.to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pretrained YOLO weights '{source}'. Provide valid local "
            "weights or allow Ultralytics to download the official weights."
        ) from exc

    model.model_name = "ultralytics_yolo"
    # Ultralytics exposes ``device`` as a read-only property; retain the
    # caller-selected inference target separately for predict().
    model.inference_device = device
    model.confidence_threshold = float(confidence_threshold)
    return model


def ensure_model(device: str, model_name: str, weights: Path | None, confidence_threshold: float):
    """Load YOLO detection model from ultralytics torch.hub or local weights."""
    torch = _import_torch()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for this detection backend but could not be imported. "
            "Verify your environment and install a compatible torch build."
        )
    try:
        if weights is not None:
            print(f"Loading local weights from {weights}")
            model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights), source="github")
        else:
            print(f"Loading YOLO model {model_name} from torch.hub")
            model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, source="github")
    except Exception as exc:
        print("Failed to load YOLO model from torch.hub.")
        print("This script expects internet access or a local weights file.")
        raise RuntimeError("YOLO model load failed") from exc

    model.to(device)
    model.conf = confidence_threshold
    return model


def detections_from_results(results: Any, model: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    if hasattr(model, "model_name") and model.model_name == "ultralytics_yolo":
        result = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        names = getattr(result, "names", getattr(model, "names", {}))
        for box, confidence, class_id in zip(xyxy, confidences, classes):
            label = str(names[class_id] if isinstance(names, dict) else names[class_id])
            if label in TRAFFIC_CLASS_NAMES and float(confidence) >= float(model.confidence_threshold):
                detections.append({
                    "xmin": float(box[0]), "ymin": float(box[1]),
                    "xmax": float(box[2]), "ymax": float(box[3]),
                    "confidence": float(confidence), "label": label,
                })
        return detections
    if hasattr(model, "model_name") and model.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        if isinstance(results, dict):
            boxes = results["boxes"].detach().cpu().numpy()
            scores = results["scores"].detach().cpu().numpy()
            labels = results["labels"].detach().cpu().numpy()
        else:
            raise RuntimeError("Unexpected torchvision detection output format")
        for box, score, label_idx in zip(boxes, scores, labels):
            if score < model.confidence_threshold:
                continue
            label = COCO_INSTANCE_CATEGORY_NAMES[int(label_idx)] if 0 <= int(label_idx) < len(COCO_INSTANCE_CATEGORY_NAMES) else str(int(label_idx))
            if label not in TRAFFIC_CLASS_NAMES:
                continue
            detections.append({
                "xmin": float(box[0]),
                "ymin": float(box[1]),
                "xmax": float(box[2]),
                "ymax": float(box[3]),
                "confidence": float(score),
                "label": label,
            })
        return detections

    if hasattr(results, "pandas"):
        try:
            df = results.pandas().xyxy[0]
            for _, row in df.iterrows():
                label = str(row["name"])
                if label not in TRAFFIC_CLASS_NAMES:
                    continue
                detections.append({
                    "xmin": float(row["xmin"]),
                    "ymin": float(row["ymin"]),
                    "xmax": float(row["xmax"]),
                    "ymax": float(row["ymax"]),
                    "confidence": float(row["confidence"]),
                    "label": label,
                })
            return detections
        except Exception:
            pass

    if hasattr(results, "xyxy"):
        try:
            tensor = results.xyxy[0].cpu().numpy()
            for row in tensor:
                label = str(model.names[int(row[5])])
                if label not in TRAFFIC_CLASS_NAMES:
                    continue
                detections.append({
                    "xmin": float(row[0]),
                    "ymin": float(row[1]),
                    "xmax": float(row[2]),
                    "ymax": float(row[3]),
                    "confidence": float(row[4]),
                    "label": label,
                })
            return detections
        except Exception:
            pass

    raise RuntimeError("Unable to parse detection output")


def run_yolo(model: Any, image: np.ndarray) -> List[Dict[str, Any]]:
    if hasattr(model, "model_name") and model.model_name == "fallback_detector":
        return model(image)
    if hasattr(model, "model_name") and model.model_name == "dummy_detector":
        return []
    if hasattr(model, "model_name") and model.model_name == "ultralytics_yolo":
        results = model.predict(
            source=image,
            conf=float(model.confidence_threshold),
            imgsz=int(getattr(model, "inference_size", 640)),
            device=model.inference_device,
            verbose=False,
        )
        return detections_from_results(results, model)
    if hasattr(model, "model_name") and model.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
        torch = _import_torch()
        transform_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform_image = torch.from_numpy(transform_image / 255.0).permute(2, 0, 1).float().to(model.device)
        with torch.no_grad():
            results = model([transform_image])[0]
        return detections_from_results(results, model)

    if hasattr(model, "model_name") and model.model_name == "dummy_detector":
        return []

    try:
        import torch
        with torch.no_grad():
            results = model(image)
    except Exception:
        results = model(image)
    return detections_from_results(results, model)


def annotate_image(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = image.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        # Read raw coords
        rx1, ry1, rx2, ry2 = det.get("xmin"), det.get("ymin"), det.get("xmax"), det.get("ymax")

        # If any coord is None, skip
        if rx1 is None or ry1 is None or rx2 is None or ry2 is None:
            continue

        # Try to robustly handle different coordinate formats:
        # - normalized coords in [0,1]
        # - swapped x/y axes
        # - inverted ranges (x1>x2 or y1>y2)
        try:
            x1f, y1f, x2f, y2f = float(rx1), float(ry1), float(rx2), float(ry2)
        except Exception:
            continue

        # Detect normalized coords (all in [0,1]) and scale to pixels
        if 0.0 <= x1f <= 1.0 and 0.0 <= x2f <= 1.0 and 0.0 <= y1f <= 1.0 and 0.0 <= y2f <= 1.0:
            x1f *= w
            x2f *= w
            y1f *= h
            y2f *= h

        # If coordinates clearly exceed image width/height in a swapped manner,
        # attempt swapping x<->y
        # e.g., xmin is > w but < h -> likely swapped
        if (x1f >= w or x2f >= w) and (x1f < h and x2f < h) and (y1f < w and y2f < w):
            # swap x and y axes
            x1f, y1f, x2f, y2f = y1f, x1f, y2f, x2f

        # Fix inverted ranges
        if x2f < x1f:
            x1f, x2f = min(x1f, x2f), max(x1f, x2f)
        if y2f < y1f:
            y1f, y2f = min(y1f, y2f), max(y1f, y2f)

        # Clamp to image bounds
        x1 = int(np.clip(x1f, 0, w - 1))
        y1 = int(np.clip(y1f, 0, h - 1))
        x2 = int(np.clip(x2f, 0, w - 1))
        y2 = int(np.clip(y2f, 0, h - 1))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        label = det.get("label", "")
        conf = float(det.get("confidence", 0.0))
        color = (0, 255, 0) if label in TRAFFIC_CLASS_NAMES else (255, 165, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return annotated


def describe_detections(detections: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(detections)
    if total == 0:
        return {
            "count": 0,
            "mean_confidence": 0.0,
            "traffic_count": 0,
            "traffic_mean_confidence": 0.0,
        }

    mean_confidence = float(np.mean([d["confidence"] for d in detections]))
    traffic = [d for d in detections if d["label"] in TRAFFIC_CLASS_NAMES]
    traffic_count = len(traffic)
    traffic_mean_confidence = float(np.mean([d["confidence"] for d in traffic])) if traffic else 0.0
    return {
        "count": total,
        "mean_confidence": mean_confidence,
        "traffic_count": traffic_count,
        "traffic_mean_confidence": traffic_mean_confidence,
    }
