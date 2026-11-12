"""Independent validation of bev_rl.energy.

Each check compares the module against a *separately derived* reference
(closed form, brute-force numerical integration, or a conservation law), not
against the module's own formula.
"""

import math
import unittest

import numpy as np

from bev_rl.energy import EnergyParams, JOULES_PER_WH, step_work_j, steady_wh_per_km, trace_energy

DT = 0.1


def brute_force_work_j(v_func, t0, t1, p: EnergyParams, n=200_001):
    """Fine-grained numerical integral of F*v dt using finite-difference acceleration."""
    t = np.linspace(t0, t1, n)
    v = v_func(t)
    a = np.gradient(v, t)
    force = p.effective_mass_kg * a + p.aero_k * v ** 2 + p.rolling_force_n
    power = force * v
    return float(np.trapezoid(power, t))


class TestEnergyValidation(unittest.TestCase):
    def setUp(self):
        self.p = EnergyParams(mass_kg=1800.0)

    def test_unit_conversion(self):
        # 1 Wh = 3600 J; 10 kW for 360 s = 1 kWh... use exact: 3600 J -> 1 Wh.
        speeds = np.full(11, 10.0)
        out = trace_energy(speeds, DT, self.p)
        self.assertAlmostEqual(out["traction_pos_wh"], out["traction_pos_j"] / JOULES_PER_WH, places=12)
        self.assertEqual(JOULES_PER_WH, 3600.0)

    def test_stationary_is_zero(self):
        out = trace_energy(np.zeros(50), DT, self.p)
        self.assertEqual(out["traction_pos_j"], 0.0)
        self.assertEqual(out["distance_m"], 0.0)
        self.assertTrue(math.isnan(out["wh_per_km"]))

    def test_constant_speed_matches_closed_form(self):
        for v in (3.0, 8.0, 20.0):
            speeds = np.full(1001, v)  # 100 s
            out = trace_energy(speeds, DT, self.p)
            expected_wh_km = (0.01 * 1800 * 9.81 + 0.5 * 1.225 * 0.23 * 2.22 * v * v) / 3.6
            self.assertAlmostEqual(out["wh_per_km"], expected_wh_km, places=9)
            self.assertAlmostEqual(out["wh_per_km"], steady_wh_per_km(v, self.p), places=9)

    def test_published_pdf_parameter_values(self):
        # Hand arithmetic from the PDF parameter table: rolling 176.6 N -> 49.06 Wh/km.
        self.assertAlmostEqual(steady_wh_per_km(0.0, self.p), 176.58 / 3.6, places=6)
        self.assertAlmostEqual(steady_wh_per_km(20.0, self.p), (176.58 + 0.3127425 * 400) / 3.6, delta=0.01)

    def test_kinetic_energy_conserved_without_resistance(self):
        p = EnergyParams(mass_kg=1500.0, drag_coefficient=1e-12, rolling_resistance=1e-12)
        speeds = np.linspace(0.0, 20.0, 101)  # accelerate to 20 m/s
        out = trace_energy(speeds, DT, p)
        self.assertAlmostEqual(out["traction_pos_j"], 0.5 * 1500.0 * 20.0 ** 2, delta=1e-3)
        self.assertAlmostEqual(out["net_work_j"], 0.5 * 1500.0 * 20.0 ** 2, delta=1e-3)

    def test_net_work_balances_kinetic_energy_plus_losses(self):
        # Independent energy balance for an arbitrary trace.
        rng = np.random.default_rng(1)
        speeds = np.clip(8 + np.cumsum(rng.normal(0, 0.15, 400)), 0.5, None)
        out = trace_energy(speeds, DT, self.p)
        dke = 0.5 * self.p.mass_kg * (speeds[-1] ** 2 - speeds[0] ** 2)
        v0, v1 = speeds[:-1], speeds[1:]
        aero = self.p.aero_k * DT * np.sum(v0 ** 3 + v0 ** 2 * v1 + v0 * v1 ** 2 + v1 ** 3) / 4.0
        roll = self.p.rolling_force_n * DT * np.sum((v0 + v1) / 2)
        self.assertAlmostEqual(out["net_work_j"], dke + aero + roll, delta=1e-6 * abs(out["net_work_j"]) + 1e-6)

    def test_matches_brute_force_integral_for_linear_ramp(self):
        v_func = lambda t: 4.0 + 3.0 * t  # linear: closed form should be exact
        exact = brute_force_work_j(v_func, 0.0, DT, self.p)
        mine = step_work_j(v_func(0.0), v_func(DT), DT, self.p)
        self.assertAlmostEqual(mine, exact, delta=1e-4 * abs(exact))

    def test_discretisation_error_on_smooth_nonlinear_trace_is_small(self):
        # Sinusoidal speed profile: sampled at 10 Hz vs brute-force integral.
        v_func = lambda t: 9.0 + 2.0 * np.sin(0.5 * t)
        horizon = 30.0
        t = np.arange(0.0, horizon + 1e-9, DT)
        truth = brute_force_work_j(v_func, 0.0, horizon, self.p)
        mine = trace_energy(v_func(t), DT, self.p)["net_work_j"]
        self.assertLess(abs(mine - truth) / abs(truth), 5e-3)

    def test_subdividing_a_linear_step_is_invariant(self):
        v0, v1 = 5.0, 9.0
        whole = step_work_j(v0, v1, DT, self.p)
        mid = 0.5 * (v0 + v1)
        halves = step_work_j(v0, mid, DT / 2, self.p) + step_work_j(mid, v1, DT / 2, self.p)
        self.assertAlmostEqual(whole, halves, delta=1e-9 * abs(whole))

    def test_braking_costs_no_traction_energy_without_regeneration(self):
        # Strong deceleration: net work is negative, so clipped energy is zero.
        out = trace_energy(np.linspace(15.0, 2.0, 40), DT, self.p)
        self.assertLess(out["net_work_j"], 0.0)
        self.assertEqual(out["traction_pos_j"], 0.0)

    def test_stop_go_cycle_costs_more_than_cruise_over_same_distance(self):
        cruise = trace_energy(np.full(201, 8.0), DT, self.p)
        accel = np.concatenate([np.linspace(0, 8, 41), np.full(60, 8.0), np.linspace(8, 0, 41)])
        stopgo = trace_energy(np.tile(accel, 3), DT, self.p)
        self.assertGreater(stopgo["wh_per_km"], steady_wh_per_km(8.0, self.p))
        self.assertAlmostEqual(cruise["wh_per_km"], steady_wh_per_km(8.0, self.p), places=9)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            step_work_j(-1.0, 1.0, DT, self.p)
        with self.assertRaises(ValueError):
            step_work_j(float("nan"), 1.0, DT, self.p)
        with self.assertRaises(ValueError):
            step_work_j(1.0, 1.0, 0.0, self.p)
        with self.assertRaises(ValueError):
            EnergyParams(mass_kg=-5.0)
        with self.assertRaises(ValueError):
            trace_energy([1.0], DT, self.p)


if __name__ == "__main__":
    unittest.main()
