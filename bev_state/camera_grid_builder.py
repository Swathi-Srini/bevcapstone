"""Camera-side BEV grid builder, version 2 (geometry and representation corrections only).

Purpose
    The repository's perception chain (five cameras -> weather filter -> front StereoSGBM depth + YOLOv8 on four views -> BEVStateAssembler.objects_from_detections) reports each detected
    vehicle as a point in the ego frame. This module converts those points into the representation the ground-truth grid uses: ONE true-size, lane-aligned rectangle per vehicle,
    centred on the vehicle, and adds the HD-map layers. It never reads simulator vehicle positions: its inputs are the detections (as converted by the unmodified perception code) and the HD map.

What the controlled-placement experiment (calibrate_camera_geometry.py, 2026-11-14) found, and what this module therefore does
    1. SIDE CAMERAS ARE POINT-REFLECTED. A vehicle on the ego's right is rendered by the camera named 'left_camera' and vice versa, while the position conversion in the unmodified
       perception code assumes the opposite orientation. For left_camera / right_camera objects the reported (x, y) is the TRUE position reflected through the ego origin (the cameras sit
       on the ego origin). In all 53 side detections of the calibration scenes the report was closer to the reflected truth than to the truth (median error 11.4 m -> 2.7 m after the reflection).
       Correction: (x, y) -> (-x, -y) for left_camera and right_camera objects.
    2. SURFACE POINT != VEHICLE CENTRE, and truncation/median-depth biases. A per-camera-group affine correction  [x, y, 1] -> centre  was fitted by least squares on placements in
       calibration scenes (seeds 2000-2002) and checked by leave-one-scene-out cross-validation (median error front 1.58 -> 0.91 m, side after reflection 2.71 -> 1.30 m; n = 52 / 33).
       The constants below are those fits. The rear camera had one sample, so it uses a geometric shift of half the footprint length along the line of sight (UNVALIDATED).
    3. TRUE-SIZE FOOTPRINT. YOLO class labels are unreliable for footprint size (a 4.6 m car is reported as 'car' or 'bus'), so every detection gets the mean true footprint of the simulator's
       traffic vehicles (2.0 m x 4.9 m), as the ground-truth grid draws the true rectangle.
    4. ONE OBJECT PER VEHICLE. Objects closer than (dedup_forward_m, dedup_lateral_m) after correction are merged by non-maximum suppression (stereo-measured objects and higher confidence win).
    5. HD-MAP LAYERS (route 0.5, boundary 0.8) are added by the same function that adds them to the simulator-derived grid.

What it does NOT do
    It cannot add a vehicle the detector did not report (YOLO misses large side-on vehicles at 4-7 m and truncated ones at the image edge, and the weather filter removes more), and it does not
    modify assembler.py, YOLO, depth, camera or weather code. Rasterisation is the ORIGINAL BEVStateAssembler.build_bev_grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from .assembler import BEVStateAssembler, ObjectFootprint

# Least-squares fits  [x_reported, y_reported, 1] -> [x_true_centre, y_true_centre]  (rows: coefficients for x_true then y_true). Fitted on calibration scenes 2000-2002 (85 placements pairs).
FRONT_AFFINE = np.array([[1.15, 0.02, -0.07], [0.11, 1.03, 0.87]])        # front_left_camera (stereo)
SIDE_AFFINE = np.array([[1.20, -0.04, -1.50], [0.02, 1.67, 0.27]])        # left_camera / right_camera, applied AFTER the point reflection
VEHICLE_WIDTH_M, VEHICLE_LENGTH_M = 2.0, 4.9                              # mean true footprint of the simulator's traffic vehicles


@dataclass(frozen=True)
class CameraBuilderConfig:
    dedup_forward_m: float = 3.5
    dedup_lateral_m: float = 1.6
    width_m: float = VEHICLE_WIDTH_M
    length_m: float = VEHICLE_LENGTH_M


class CameraGridBuilder:
    def __init__(self, assembler: BEVStateAssembler | None = None,
                 overlay: Callable[[np.ndarray, object], np.ndarray] | None = None,
                 config: CameraBuilderConfig | None = None):
        self.asm = assembler or BEVStateAssembler()
        self.cfg = config or CameraBuilderConfig()
        self._overlay = overlay

    # ---------------------------------------------------------------- corrections
    def correct(self, obj: ObjectFootprint) -> ObjectFootprint:
        x, y = obj.x_right_m, obj.y_forward_m
        if obj.camera in ("left_camera", "right_camera"):
            x, y = -x, -y                                                   # correction 1: point reflection of the side cameras
            x, y = (SIDE_AFFINE @ np.array([x, y, 1.0])).tolist()           # correction 2 (fitted)
        elif obj.camera == "front_left_camera":
            x, y = (FRONT_AFFINE @ np.array([x, y, 1.0])).tolist()
        elif obj.camera == "rear_camera":                                   # unvalidated geometric shift: the near (front) face of a vehicle behind is seen
            rho = math.hypot(x, y)
            if rho > 1e-6:
                s = min((self.cfg.width_m / 2) / max(abs(x / rho), 1e-3), (self.cfg.length_m / 2) / max(abs(y / rho), 1e-3)); x, y = x + x / rho * s, y + y / rho * s
        return replace(obj, x_right_m=float(x), y_forward_m=float(y), width_m=self.cfg.width_m, length_m=self.cfg.length_m)   # correction 3

    def merge_duplicates(self, objects: Sequence[ObjectFootprint]) -> list[ObjectFootprint]:
        """Correction 4: non-maximum suppression in the BEV plane."""
        kept: list[ObjectFootprint] = []
        for o in sorted(objects, key=lambda o: (o.depth_method == "stereo_sgm", o.confidence), reverse=True):
            if all(abs(o.y_forward_m - k.y_forward_m) >= self.cfg.dedup_forward_m or abs(o.x_right_m - k.x_right_m) >= self.cfg.dedup_lateral_m for k in kept):
                kept.append(o)
        return kept

    def corrected_objects(self, objects: Sequence[ObjectFootprint]) -> list[ObjectFootprint]:
        return self.merge_duplicates([self.correct(o) for o in objects])

    # ---------------------------------------------------------------- grid
    def build(self, objects: Sequence[ObjectFootprint], md=None) -> np.ndarray:
        """Camera-derived grid: corrected objects rasterised by the ORIGINAL builder, then the HD-map layers (route 0.5, boundary 0.8)."""
        grid = self.asm.build_bev_grid(self.corrected_objects(objects))
        if md is not None:
            overlay = self._overlay
            if overlay is None:
                from bev_rl.adaptive.spec_bev import overlay_github_bev as overlay        # same function that adds the map layers to the simulator-derived grid
            grid = overlay(grid, md)
        return grid
