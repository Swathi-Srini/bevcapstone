"""Metrics shared by BEV-policy closed-loop evaluation scripts.

The action-delta metric is deliberately called a *control-change proxy* rather
than physical jerk: MetaDrive's action is a normalized steering/longitudinal
command, not measured vehicle acceleration.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def action_metrics(actions: Sequence[np.ndarray]) -> dict[str, float]:
    """Return interpretable control statistics for one episode."""
    if not actions:
        return {
            "mean_abs_steering": 0.0,
            "mean_throttle_command": 0.0,
            "mean_brake_command": 0.0,
            "mean_abs_longitudinal_delta": 0.0,
        }
    action_array = np.asarray(actions, dtype=np.float32)
    longitudinal = action_array[:, 1]
    deltas = np.diff(longitudinal) if len(longitudinal) > 1 else np.zeros(1, dtype=np.float32)
    return {
        "mean_abs_steering": float(np.mean(np.abs(action_array[:, 0]))),
        "mean_throttle_command": float(np.mean(np.clip(longitudinal, 0.0, None))),
        "mean_brake_command": float(np.mean(np.clip(-longitudinal, 0.0, None))),
        "mean_abs_longitudinal_delta": float(np.mean(np.abs(deltas))),
    }


def summarise_episode_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate paired closed-loop episode rows without hiding failures."""
    if not rows:
        return {"episodes": 0, "success_rate": 0.0, "collision_rate": 0.0, "off_road_rate": 0.0,
                "mean_speed_mps": 0.0, "mean_steps": 0.0, "mean_abs_steering": 0.0,
                "mean_abs_longitudinal_delta": 0.0}

    def mean(key: str) -> float:
        return float(np.mean([float(row.get(key, 0.0)) for row in rows]))

    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
        "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
        "off_road_rate": float(np.mean([row["outcome"] == "off_road" for row in rows])),
        "mean_speed_mps": mean("mean_speed_mps"),
        "mean_steps": mean("steps"),
        "mean_abs_steering": mean("mean_abs_steering"),
        "mean_abs_longitudinal_delta": mean("mean_abs_longitudinal_delta"),
    }
