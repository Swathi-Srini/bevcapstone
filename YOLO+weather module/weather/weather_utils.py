"""Reproducible, physically motivated camera-weather augmentations.

These are image-formation approximations, not a replacement for a weather
renderer or measured sensor data. Fog uses atmospheric light and
Beer-Lambert transmission over a ground-plane depth prior; rain adds exposure
streaks, distance-dependent veiling, wet-road darkening, and lens droplets.
Policy pipelines use measured front stereo depth where valid and the prior
elsewhere.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _random_source(rng: Any | None) -> Any:
    return np.random if rng is None else rng


def _integers(random: Any, low: int, high: int, size: Any = None) -> Any:
    if hasattr(random, "integers"):
        return random.integers(low, high, size=size)
    return random.randint(low, high, size=size)


def ground_plane_depth_prior(
    image_shape: tuple[int, int],
    *,
    camera_height_m: float = 1.4,
    downward_pitch_deg: float = 5.0,
    horizontal_fov_deg: float = 60.0,
) -> np.ndarray:
    """Estimate a coarse per-row ground range, saturating sky/horizon at 150 m."""

    height, width = image_shape
    focal_length = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg / 2.0))
    rows = np.arange(height, dtype=np.float32)
    ray_angle = math.radians(downward_pitch_deg) + np.arctan(
        (rows - height / 2.0) / focal_length
    )
    depth = np.full(height, 150.0, dtype=np.float32)
    ground = ray_angle > math.radians(0.25)
    depth[ground] = camera_height_m / np.tan(ray_angle[ground])
    return np.broadcast_to(np.clip(depth, 2.0, 150.0)[:, None], (height, width)).copy()


def _depth_map(image: np.ndarray, depth_map_m: np.ndarray | None) -> np.ndarray:
    if depth_map_m is None:
        return ground_plane_depth_prior(image.shape[:2])
    depth = np.asarray(depth_map_m, dtype=np.float32)
    if depth.shape != image.shape[:2]:
        raise ValueError(
            f"depth_map_m must have shape {image.shape[:2]}, received {depth.shape}"
        )
    prior = ground_plane_depth_prior(image.shape[:2])
    valid = np.isfinite(depth) & (depth > 0.0)
    depth = np.where(valid, depth, prior)
    return np.clip(depth, 2.0, 150.0)


def _atmospheric_attenuation(
    image: np.ndarray,
    intensity: float,
    visibility_m: float,
    depth_map_m: np.ndarray | None,
    airlight: tuple[int, int, int] = (220, 228, 235),
) -> np.ndarray:
    depth = _depth_map(image, depth_map_m)
    beta = -math.log(0.05) / max(visibility_m, 1.0)
    transmission = np.exp(-beta * depth).astype(np.float32)[..., None]
    source = image.astype(np.float32)
    atmosphere = np.asarray(airlight, dtype=np.float32).reshape(1, 1, 3)
    veiled = source * transmission + atmosphere * (1.0 - transmission)
    return np.clip(source * (1.0 - intensity) + veiled * intensity, 0, 255).astype(np.uint8)


def add_fog(
    image: np.ndarray,
    intensity: float,
    *,
    depth_map_m: np.ndarray | None = None,
) -> np.ndarray:
    """Apply depth-dependent atmospheric scattering with user-scaled intensity."""

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity == 0.0:
        return image.copy()
    # Clear-to-dense control: 0 -> effectively clear at simulator distances;
    # 1 -> roughly 9 m meteorological visibility under this image model.
    visibility_m = 2.0 + 118.0 * math.exp(-2.8 * intensity)
    return _atmospheric_attenuation(image, intensity, visibility_m, depth_map_m)


def add_rain(
    image: np.ndarray,
    intensity: float,
    *,
    depth_map_m: np.ndarray | None = None,
    rng: Any | None = None,
) -> np.ndarray:
    """Render a controlled rain-rate proxy and its camera-visible effects."""

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity == 0.0:
        return image.copy()
    random = _random_source(rng)
    height, width = image.shape[:2]

    # Rain rate is a control mapping for repeatable experiments, not a
    # calibrated millimetres-per-hour sensor model.
    rain_rate_mm_h = 2.0 + 78.0 * intensity
    visibility_m = 120.0 / (1.0 + 0.035 * rain_rate_mm_h)
    rainy = _atmospheric_attenuation(
        image, 0.28 * intensity, visibility_m, depth_map_m, airlight=(205, 215, 225)
    )

    # Wet-road darkening is limited to the lower image, where a flat-road
    # camera model places the road surface; rain streaks are alpha-composited
    # with varied length, angle, and exposure rather than hard white lines.
    row_mask = np.linspace(0.0, 1.0, height, dtype=np.float32)
    road_weight = np.clip((row_mask - 0.48) / 0.30, 0.0, 1.0)[:, None, None]
    darkening = 1.0 - 0.18 * intensity * road_weight
    rainy = np.clip(rainy.astype(np.float32) * darkening, 0, 255).astype(np.uint8)

    streaks = np.zeros_like(rainy)
    count = int((height * width / 30_000.0) * (2.0 + 16.0 * intensity))
    for _ in range(count):
        x1 = int(_integers(random, 0, width))
        y1 = int(_integers(random, 0, height))
        length = int(_integers(random, max(2, height // 80), max(3, height // 18)))
        dx = int(_integers(random, -max(1, length // 5), max(2, length // 5 + 1)))
        y2 = min(height - 1, y1 + length)
        color = int(_integers(random, 175, 246))
        thickness = int(_integers(random, 1, 3))
        cv2.line(streaks, (x1, y1), (x1 + dx, y2), (color, color, color),
                 thickness, cv2.LINE_AA)
    streak_alpha = 0.08 + 0.24 * intensity
    rainy = cv2.addWeighted(rainy, 1.0, streaks, streak_alpha, 0.0)

    # Large droplets on the virtual lens create localized soft occlusions.
    droplet_layer = rainy.copy()
    droplet_count = int(2 + 22 * intensity)
    for _ in range(droplet_count):
        radius = int(_integers(random, 2, max(3, min(height, width) // 35 + 1)))
        x, y = int(_integers(random, 0, width)), int(_integers(random, 0, height))
        shade = int(_integers(random, 145, 226))
        cv2.circle(droplet_layer, (x, y), radius, (shade, shade, shade), -1, cv2.LINE_AA)
        cv2.ellipse(droplet_layer, (x, y), (radius, max(1, radius // 2)), 0,
                    200, 340, (245, 245, 245), 1, cv2.LINE_AA)
    droplet_layer = cv2.GaussianBlur(droplet_layer, (3, 3), 0.7)
    return cv2.addWeighted(rainy, 1.0 - 0.12 * intensity,
                           droplet_layer, 0.12 * intensity, 0.0)


def weather_visibility_horizon(weather: str, intensity: float) -> float | None:
    """Return a conservative 50%-contrast horizon for controlled BEV studies.

    It is derived from the same analytic image model, not measured from a
    camera and not a claim about MetaDrive's native rendering.
    """

    intensity = float(np.clip(intensity, 0.0, 1.0))
    if weather in ("none", "clear") or intensity == 0.0:
        return None
    if weather == "fog":
        visibility_m = 2.0 + 118.0 * math.exp(-2.8 * intensity)
    elif weather == "rain":
        rain_rate_mm_h = 2.0 + 78.0 * intensity
        visibility_m = 120.0 / (1.0 + 0.035 * rain_rate_mm_h)
    elif weather == "all":
        fog_visibility = 2.0 + 118.0 * math.exp(-2.8 * intensity)
        rain_rate_mm_h = 2.0 + 78.0 * intensity
        rain_visibility = 120.0 / (1.0 + 0.035 * rain_rate_mm_h)
        visibility_m = min(fog_visibility, rain_visibility)
    else:
        raise ValueError(f"Unknown weather mode: {weather}")
    return float(np.clip(0.231 * visibility_m, 2.0, 30.0))


def apply_weather(
    image: np.ndarray,
    weather: str,
    level: float,
    *,
    depth_map_m: np.ndarray | None = None,
    rng: Any | None = None,
) -> np.ndarray:
    """Apply ``none``/``clear``, fog, rain, or combined synthetic weather."""

    if not 0.0 <= float(level) <= 1.0:
        raise ValueError("Weather intensity must be between 0 and 1.")
    if weather in ("none", "clear"):
        return image.copy()
    if weather == "fog":
        return add_fog(image, level, depth_map_m=depth_map_m)
    if weather == "rain":
        return add_rain(image, level, depth_map_m=depth_map_m, rng=rng)
    if weather == "all":
        fogged = add_fog(image, level, depth_map_m=depth_map_m)
        return add_rain(fogged, level, depth_map_m=depth_map_m, rng=rng)
    raise ValueError(f"Unknown weather mode: {weather}")


def prepare_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 4:
        image = image[..., -1]
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image
