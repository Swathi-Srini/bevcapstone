"""Shared weather-to-camera-to-YOLO path for collection and evaluation."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from manual_drive_stereo_yolo_weather import (
    CAMERA_RIGS,
    YOLO_CAMERAS,
    apply_synchronized_stereo_weather,
    front_stereo_depth,
)
from weather.weather_utils import apply_weather, weather_visibility_horizon
from yolo.yolo_utils import run_yolo_batch


def run_perception(
    frames: Mapping[str, np.ndarray],
    yolo_model: Any,
    weather: str,
    intensity: float,
    rng: np.random.Generator,
    stereo_matcher: Any | None = None,
    stereo_scale: float = 1.0,
    return_depth: bool = False,
) -> tuple:
    """Apply weather, one shared stereo pass, and YOLO to the camera inputs."""

    if weather in ("none", "clear") or intensity == 0.0:
        conditioned = {name: image.copy() for name, image in frames.items()}
    else:
        conditioned = {}
        left, right = apply_synchronized_stereo_weather(
            frames["front_left_camera"],
            frames["front_right_camera"],
            weather,
            intensity,
            rng=rng,
            # Weather rendering uses its ground-plane depth prior. The policy
            # depth below is computed separately from the degraded stereo pair.
            depth_map_m=None,
        )
        conditioned["front_left_camera"], conditioned["front_right_camera"] = left, right
        for name in CAMERA_RIGS:
            if name not in ("front_left_camera", "front_right_camera"):
                conditioned[name] = apply_weather(
                    frames[name], weather, intensity, rng=rng
                )

    # This is the depth actually available to the policy, computed once from
    # the camera pair after any weather degradation.
    front_depth = None
    if stereo_matcher is not None:
        front_depth = front_stereo_depth(
            conditioned["front_left_camera"], conditioned["front_right_camera"],
            stereo_matcher, scale=stereo_scale,
        )

    batched_detections = run_yolo_batch(
        yolo_model, [conditioned[name] for name in YOLO_CAMERAS]
    )
    detections = dict(zip(YOLO_CAMERAS, batched_detections))
    if return_depth:
        return conditioned, detections, weather_visibility_horizon(weather, intensity), front_depth
    return conditioned, detections, weather_visibility_horizon(weather, intensity)
