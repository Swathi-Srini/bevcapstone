"""Weather augmentation helpers for fog and rain effects."""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any


def add_fog(image: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0.0:
        return image.copy()
    intensity = float(np.clip(intensity, 0.0, 1.0))
    fog_color = np.full_like(image, 255, dtype=np.uint8)
    h, w = image.shape[:2]
    alpha = np.linspace(0.0, intensity * 0.9, h, dtype=np.float32)[:, None, None]
    fogged = image.astype(np.float32) * (1.0 - alpha) + fog_color.astype(np.float32) * alpha
    fogged = np.clip(fogged, 0, 255).astype(np.uint8)
    ksize = max(3, int(15 * intensity) | 1)
    fogged = cv2.GaussianBlur(fogged, (ksize, ksize), sigmaX=8 * intensity + 1)
    noise = np.random.normal(0, 4.0 * intensity, size=fogged.shape).astype(np.float32)
    return np.clip(fogged.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_rain(image: np.ndarray, intensity: float) -> np.ndarray:
    if intensity <= 0.0:
        return image.copy()
    intensity = float(np.clip(intensity, 0.0, 1.0))
    h, w = image.shape[:2]
    rain_layer = np.zeros_like(image, dtype=np.uint8)
    line_count = int(800 * intensity)
    for _ in range(line_count):
        x1 = np.random.randint(-w // 2, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(max(1, h // 20), max(2, h // 8))
        x2 = int(x1 + length * 0.2)
        y2 = min(h - 1, y1 + length)
        color = int(np.random.randint(180, 245))
        thickness = np.random.randint(1, 2 + int(2 * intensity))
        cv2.line(rain_layer, (x1, y1), (x2, y2), (color, color, color), thickness)
    rain_layer = cv2.blur(rain_layer, (3, 3))
    base = image.astype(np.uint8)
    rainy = cv2.addWeighted(base, 1.0 - 0.1 * intensity, rain_layer, 0.2 + 0.4 * intensity, 0)
    kernel = np.array([[0, 0.25, 0], [0, 0.25, 0], [0, 0.25, 0]], dtype=np.float32)
    rainy = cv2.filter2D(rainy, -1, kernel)
    return np.clip(rainy, 0, 255).astype(np.uint8)


def apply_weather(image: np.ndarray, weather: str, level: float) -> np.ndarray:
    if weather == "none":
        return image.copy()
    if weather == "fog":
        return add_fog(image, level)
    if weather == "rain":
        return add_rain(image, level)
    if weather == "all":
        return add_rain(add_fog(image, level), level)
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
