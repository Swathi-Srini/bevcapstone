"""Estimated mechanical traction energy from simulated longitudinal speed.

SCOPE (read before quoting any number produced here)
----------------------------------------------------
* This is an *estimate of mechanical work at the wheels* for a declared virtual
  road-vehicle model, applied to the speed trace produced by MetaDrive. It is
  NOT battery electricity and NOT a MetaDrive-internal quantity: MetaDrive's
  Bullet vehicle has no aerodynamic drag and its built-in ``energy_consumption``
  is a fuel model, so the drag/rolling terms below are *modelled*, not measured.
* Regenerative braking, drivetrain efficiency and auxiliary loads are NOT
  modelled. ``traction_pos_j`` is therefore the traction energy with zero
  regeneration (the PDF's ``max(0, P*dt)``).
* Road gradient is zero (MetaDrive procedurally generated maps are flat).

Per decision step of length ``dt`` the speed is assumed to vary linearly from
``v0`` to ``v1`` (the only information available at the 10 Hz decision rate).
Under that assumption the work integral is evaluated in closed form:

    inertial :  m_eff * (v1^2 - v0^2) / 2                  (= change in kinetic energy)
    aero     :  k * dt * (v0^3 + v0^2 v1 + v0 v1^2 + v1^3) / 4,  k = 0.5 rho Cd A
    rolling  :  Crr * m * g * (v0 + v1)/2 * dt

so a step's net work is exact for linear speed, and kinetic energy is conserved
by construction. ``traction_pos_j`` clips each step's net work at zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

JOULES_PER_WH = 3600.0


@dataclass(frozen=True)
class EnergyParams:
    """Declared virtual-vehicle parameters. All must be positive except grade."""

    mass_kg: float                     # taken from the simulator vehicle at run time
    drag_coefficient: float = 0.23     # PDF Table 11 (declared assumption)
    frontal_area_m2: float = 2.22      # PDF Table 11 (declared assumption)
    rolling_resistance: float = 0.01   # PDF Table 11 (declared assumption)
    air_density_kg_m3: float = 1.225
    gravity_mps2: float = 9.81
    rotational_mass_kg: float = 0.0    # PDF 4I/r^2 term; 0.0 = omitted, as in the PDF code

    def __post_init__(self) -> None:
        for name in ("mass_kg", "drag_coefficient", "frontal_area_m2",
                     "rolling_resistance", "air_density_kg_m3", "gravity_mps2"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not np.isfinite(self.rotational_mass_kg) or self.rotational_mass_kg < 0.0:
            raise ValueError("rotational_mass_kg must be finite and non-negative")

    @property
    def effective_mass_kg(self) -> float:
        return self.mass_kg + self.rotational_mass_kg

    @property
    def aero_k(self) -> float:
        """0.5 * rho * Cd * A  (N per (m/s)^2)."""
        return 0.5 * self.air_density_kg_m3 * self.drag_coefficient * self.frontal_area_m2

    @property
    def rolling_force_n(self) -> float:
        return self.rolling_resistance * self.mass_kg * self.gravity_mps2


def step_work_j(v0: float, v1: float, dt: float, params: EnergyParams) -> float:
    """Signed net mechanical work (J) needed over one step, linear-speed closed form."""
    if not (np.isfinite(v0) and np.isfinite(v1) and np.isfinite(dt)):
        raise ValueError("speeds and dt must be finite")
    if v0 < 0.0 or v1 < 0.0:
        raise ValueError("speeds must be non-negative (longitudinal speed magnitude)")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    inertial = 0.5 * params.effective_mass_kg * (v1 * v1 - v0 * v0)
    aero = params.aero_k * dt * (v0 ** 3 + v0 * v0 * v1 + v0 * v1 * v1 + v1 ** 3) / 4.0
    rolling = params.rolling_force_n * 0.5 * (v0 + v1) * dt
    return float(inertial + aero + rolling)


def trace_energy(speeds_mps, dt: float, params: EnergyParams) -> dict:
    """Energy summary for a speed trace sampled every ``dt`` seconds.

    ``speeds_mps`` includes the initial speed, so N+1 samples describe N steps.
    Returns joules, Wh, distance and the per-step arrays.
    """
    speeds = np.asarray(speeds_mps, dtype=np.float64)
    if speeds.ndim != 1 or speeds.size < 2:
        raise ValueError("speeds_mps must be 1-D with at least two samples")
    work = np.array([step_work_j(a, b, dt, params) for a, b in zip(speeds[:-1], speeds[1:])])
    positive = np.clip(work, 0.0, None)
    distance_m = float(np.sum(0.5 * (speeds[:-1] + speeds[1:]) * dt))
    pos_j = float(positive.sum())
    return {
        "traction_pos_j": pos_j,
        "traction_pos_wh": pos_j / JOULES_PER_WH,
        "net_work_j": float(work.sum()),
        "distance_m": distance_m,
        "wh_per_km": (pos_j / JOULES_PER_WH) / (distance_m / 1000.0) if distance_m > 1e-9 else float("nan"),
        "step_work_j": work,
        "step_pos_j": positive,
    }


def steady_wh_per_km(speed_mps: float, params: EnergyParams) -> float:
    """Closed-form constant-speed traction energy, for cross-checking."""
    force = params.rolling_force_n + params.aero_k * speed_mps ** 2
    return force / 3.6
