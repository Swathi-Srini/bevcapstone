"""Per-step vehicle telemetry read from MetaDrive (training/evaluation signals only).

These values are used for the reward and the metrics. None of them is fed to the
actor except through the already-defined 6-D scalar state built by BEVStateAssembler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Telemetry:
    speed_mps: float
    lateral_m: float
    heading_err_rad: float
    route_completion: float
    route_length_m: float
    x: float
    y: float

    @property
    def progress_m(self) -> float:
        return self.route_completion * self.route_length_m


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_telemetry(env) -> Telemetry:
    agent = env.agent
    nav = agent.navigation
    speed = float(agent.speed)                      # MetaDrive BaseObject.speed is m/s (verified in source)
    lateral, heading = 0.0, 0.0
    lane = getattr(agent, "lane", None)
    if lane is not None:
        s, lateral = lane.local_coordinates(agent.position)
        heading = _wrap(float(agent.heading_theta) - float(lane.heading_theta_at(s)))
    pos = np.asarray(agent.position, dtype=np.float64)
    return Telemetry(
        speed_mps=speed, lateral_m=float(lateral), heading_err_rad=float(heading),
        route_completion=float(np.clip(nav.route_completion, 0.0, 1.0)),
        route_length_m=float(nav.total_length), x=float(pos[0]), y=float(pos[1]),
    )


def vehicle_mass_kg(env) -> float:
    """Simulator vehicle mass, read from the live agent (never assumed)."""
    agent = env.agent
    mass = agent.config.get("mass") if hasattr(agent, "config") else None
    if not mass:
        mass = getattr(agent, "MASS", None)
    if not mass:
        mass = float(agent.body.getMass())
    return float(mass)
