"""Assemble policy-ready BEV and ego-state observations.

This module is the bridge between the existing MetaDrive perception prototypes
and the next person's RL training code. It intentionally avoids training logic.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION_ROOT = PROJECT_ROOT / "YOLO+weather module"
if str(PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_ROOT))

from manual_drive_stereo_yolo_weather import (  # noqa: E402
    CAMERA_RIGS,
    CAM_FOV,
    CAM_H,
    CAM_W,
    YOLO_CAMERAS,
    build_stereo_matcher,
    estimate_detection_position,
    frame_from_sensor,
    front_stereo_depth,
    mount_camera_rig,
)
from yolo.yolo_utils import run_yolo  # noqa: E402


@dataclass(frozen=True)
class BEVValues:
    """Numeric BEV encoding used by downstream policy code."""

    unknown: float = -1.0
    free: float = 0.0
    route: float = 0.5
    occupied: float = 1.0
    ego: float = 0.9


@dataclass(frozen=True)
class BEVStateConfig:
    """Geometry and scaling for the final 64x64 state grid."""

    grid_size: int = 64
    lateral_range_m: float = 10.0
    forward_range_m: float = 17.5
    rear_range_m: float = 2.5
    camera_range_m: float = 30.0
    ego_width_m: float = 1.9
    ego_length_m: float = 4.6
    values: BEVValues = BEVValues()

    @property
    def metres_per_cell(self) -> float:
        return (2.0 * self.lateral_range_m) / float(self.grid_size)

    @property
    def total_forward_extent_m(self) -> float:
        return self.forward_range_m + self.rear_range_m


@dataclass(frozen=True)
class ObjectFootprint:
    """A detected object projected into ego-frame metres."""

    label: str
    x_right_m: float
    y_forward_m: float
    width_m: float
    length_m: float
    confidence: float = 1.0
    camera: str = "unknown"
    depth_method: str = "unknown"


@dataclass(frozen=True)
class StateObservation:
    """Policy input assembled at one simulator step."""

    bev_grid: np.ndarray
    scalar_state: np.ndarray
    objects: tuple[ObjectFootprint, ...]
    frames: Mapping[str, np.ndarray] | None = None
    front_depth: np.ndarray | None = None


class BEVStateAssembler:
    """Build the final BEV grid and 6-D ego state.

    Scalar state order:
        [speed_mps, acceleration_mps2, steering, heading_error_rad,
         lane_offset_m, route_completion]
    """

    CLASS_FOOTPRINTS_M = {
        "person": (0.8, 0.8),
        "car": (1.9, 4.6),
        "motorcycle": (0.8, 2.2),
        "bus": (2.6, 12.0),
        "truck": (2.6, 8.0),
    }

    def __init__(self, config: BEVStateConfig | None = None):
        self.config = config or BEVStateConfig()
        self.matcher = build_stereo_matcher()
        self._mounted_env_ids: set[int] = set()

    def assemble(
        self,
        env: Any | None = None,
        yolo_model: Any | None = None,
        info: Mapping[str, Any] | None = None,
        detections_by_camera: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
        frames: Mapping[str, np.ndarray] | None = None,
    ) -> StateObservation:
        """Return a policy-ready observation.

        Pass an active MetaDrive env and YOLO model for live assembly. For unit
        tests or offline use, pass detections_by_camera directly.
        """

        live_frames = dict(frames) if frames is not None else None
        front_depth = None

        if env is not None and live_frames is None:
            self.mount_if_needed(env)
            live_frames = {name: frame_from_sensor(env, name) for name in CAMERA_RIGS}

        if live_frames is not None and {"front_left_camera", "front_right_camera"} <= set(live_frames):
            front_depth = front_stereo_depth(
                live_frames["front_left_camera"],
                live_frames["front_right_camera"],
                self.matcher,
            )

        if detections_by_camera is None:
            detections_by_camera = {}
            if yolo_model is not None and live_frames is not None:
                detections_by_camera = {
                    name: run_yolo(yolo_model, live_frames[name])
                    for name in YOLO_CAMERAS
                }

        objects = self.objects_from_detections(detections_by_camera, front_depth)
        bev_grid = self.build_bev_grid(objects)
        scalar_state = self.extract_scalar_state(env, info)

        return StateObservation(
            bev_grid=bev_grid,
            scalar_state=scalar_state,
            objects=tuple(objects),
            frames=live_frames,
            front_depth=front_depth,
        )

    def mount_if_needed(self, env: Any) -> None:
        """Attach the five-camera rig once per environment object."""

        env_id = id(env)
        if env_id not in self._mounted_env_ids:
            mount_camera_rig(env)
            self._mounted_env_ids.add(env_id)

    def objects_from_detections(
        self,
        detections_by_camera: Mapping[str, Sequence[Mapping[str, Any]]],
        front_depth: np.ndarray | None,
    ) -> list[ObjectFootprint]:
        """Convert YOLO boxes into ego-frame object footprints."""

        objects: list[ObjectFootprint] = []
        depth = front_depth if front_depth is not None else np.zeros((CAM_H, CAM_W), dtype=np.float32)

        for camera, detections in detections_by_camera.items():
            for detection in detections:
                method, depth_m, x_right_m, y_forward_m = estimate_detection_position(camera, dict(detection), depth)
                if depth_m is None or x_right_m is None or y_forward_m is None:
                    continue
                width_m, length_m = self.estimate_footprint(dict(detection), depth_m)
                objects.append(ObjectFootprint(
                    label=str(detection.get("label", "unknown")),
                    x_right_m=float(x_right_m),
                    y_forward_m=float(y_forward_m),
                    width_m=float(width_m),
                    length_m=float(length_m),
                    confidence=float(detection.get("confidence", 1.0)),
                    camera=camera,
                    depth_method=method,
                ))

        return objects

    def estimate_footprint(self, detection: Mapping[str, Any], depth_m: float) -> tuple[float, float]:
        """Estimate physical width/length for a detected object.

        COCO classes do not provide object orientation, so class priors are the
        stable footprint contract. Bbox-derived width is used only as a sanity
        clamp for unusual scales.
        """

        label = str(detection.get("label", "car"))
        prior_width, prior_length = self.CLASS_FOOTPRINTS_M.get(label, (1.9, 4.6))

        try:
            bbox_width_px = float(detection["xmax"]) - float(detection["xmin"])
            apparent_width_m = bbox_width_px * float(depth_m) / 1000.0
            if math.isfinite(apparent_width_m) and apparent_width_m > 0:
                prior_width = float(np.clip(apparent_width_m, prior_width * 0.6, prior_width * 1.6))
        except (KeyError, TypeError, ValueError):
            pass

        return prior_width, prior_length

    def build_bev_grid(self, objects: Sequence[ObjectFootprint]) -> np.ndarray:
        """Build a 64x64 grid with unknown/free/occupied/ego semantics."""

        cfg = self.config
        grid = np.full((cfg.grid_size, cfg.grid_size), cfg.values.unknown, dtype=np.float32)
        self._mark_visible_free_space(grid)

        for obj in objects:
            self._draw_oriented_footprint(
                grid,
                obj.x_right_m,
                obj.y_forward_m,
                obj.width_m,
                obj.length_m,
                cfg.values.occupied,
            )

        self._draw_oriented_footprint(grid, 0.0, 0.0, cfg.ego_width_m, cfg.ego_length_m, cfg.values.ego)
        return grid

    def ego_to_grid(self, x_right_m: float, y_forward_m: float) -> tuple[int, int] | None:
        """Convert ego-frame metres to integer (row, col), or None if outside."""

        cfg = self.config
        col = int((x_right_m + cfg.lateral_range_m) / (2.0 * cfg.lateral_range_m) * cfg.grid_size)
        row = int((1.0 - (y_forward_m + cfg.rear_range_m) / cfg.total_forward_extent_m) * cfg.grid_size)
        if 0 <= row < cfg.grid_size and 0 <= col < cfg.grid_size:
            return row, col
        return None

    def _mark_visible_free_space(self, grid: np.ndarray) -> None:
        """Mark cells seen by the four logical camera views as free."""

        cfg = self.config
        half_fov = math.radians(CAM_FOV / 2.0)
        y_values = np.linspace(cfg.forward_range_m, -cfg.rear_range_m, cfg.grid_size)
        x_values = np.linspace(-cfg.lateral_range_m, cfg.lateral_range_m, cfg.grid_size)
        camera_yaws = {
            "front_left_camera": 0.0,
            "left_camera": -math.pi / 2.0,
            "right_camera": math.pi / 2.0,
            "rear_camera": math.pi,
        }

        for row, y_forward_m in enumerate(y_values):
            for col, x_right_m in enumerate(x_values):
                distance = math.hypot(x_right_m, y_forward_m)
                if distance > cfg.camera_range_m:
                    continue
                angle = math.atan2(x_right_m, y_forward_m)
                if any(abs(self._angle_diff(angle, yaw)) <= half_fov for yaw in camera_yaws.values()):
                    grid[row, col] = cfg.values.free

    def _draw_oriented_footprint(
        self,
        grid: np.ndarray,
        x_right_m: float,
        y_forward_m: float,
        width_m: float,
        length_m: float,
        value: float,
    ) -> None:
        """Draw a footprint scaled from physical metres into grid cells."""

        cfg = self.config
        center = self.ego_to_grid(x_right_m, y_forward_m)
        if center is None:
            return

        half_width_cells = max(1, int(math.ceil(width_m / cfg.metres_per_cell / 2.0)))
        half_length_cells = max(1, int(math.ceil(length_m / cfg.metres_per_cell / 2.0)))
        center_row, center_col = center

        row_min = max(0, center_row - half_length_cells)
        row_max = min(cfg.grid_size - 1, center_row + half_length_cells)
        col_min = max(0, center_col - half_width_cells)
        col_max = min(cfg.grid_size - 1, center_col + half_width_cells)
        grid[row_min:row_max + 1, col_min:col_max + 1] = value

    def extract_scalar_state(self, env: Any | None, info: Mapping[str, Any] | None = None) -> np.ndarray:
        """Extract the 6-D ego state expected by the BEV policy."""

        info = info or {}
        speed_mps = self._speed_mps(env, info)
        acceleration = self._scalar_from_info(info, "acceleration", default=0.0)
        steering = self._scalar_from_info(info, "steering", default=self._agent_scalar(env, "steering", 0.0))
        route_completion = float(np.clip(self._scalar_from_info(info, "route_completion", default=0.0), 0.0, 1.0))
        heading_error = self._heading_error(env)
        lane_offset = self._lane_offset(env)
        return np.asarray(
            [speed_mps, acceleration, steering, heading_error, lane_offset, route_completion],
            dtype=np.float32,
        )

    def _speed_mps(self, env: Any | None, info: Mapping[str, Any]) -> float:
        velocity = info.get("velocity")
        if velocity is not None:
            arr = np.asarray(velocity, dtype=np.float32)
            if arr.size > 1:
                return float(np.linalg.norm(arr))
            return float(arr.reshape(-1)[0])
        agent = getattr(env, "agent", None)
        speed = getattr(agent, "speed", None)
        if speed is not None:
            return float(speed)
        return float(getattr(agent, "speed_km_h", 0.0)) / 3.6

    def _heading_error(self, env: Any | None) -> float:
        agent = getattr(env, "agent", None)
        lane = getattr(agent, "lane", None)
        if agent is None or lane is None:
            return 0.0
        try:
            longitudinal, _ = lane.local_coordinates(agent.position)
            lane_heading = lane.heading_theta_at(longitudinal)
            return self._angle_diff(float(getattr(agent, "heading_theta", 0.0)), float(lane_heading))
        except Exception:
            return 0.0

    def _lane_offset(self, env: Any | None) -> float:
        agent = getattr(env, "agent", None)
        lane = getattr(agent, "lane", None)
        if agent is None or lane is None:
            return 0.0
        try:
            _, lateral = lane.local_coordinates(agent.position)
            return float(lateral)
        except Exception:
            return 0.0

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _scalar_from_info(info: Mapping[str, Any], key: str, default: float = 0.0) -> float:
        value = info.get(key, default)
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _agent_scalar(env: Any | None, key: str, default: float = 0.0) -> float:
        agent = getattr(env, "agent", None)
        value = getattr(agent, key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
