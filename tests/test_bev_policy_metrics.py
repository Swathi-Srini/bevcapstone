import unittest

import numpy as np

from bev_policy.metrics import action_metrics, summarise_episode_rows


class TestBEVPolicyMetrics(unittest.TestCase):
    def test_action_metrics_split_throttle_and_brake(self):
        metrics = action_metrics([np.array([0.2, 0.5]), np.array([-0.4, -0.25])])
        self.assertAlmostEqual(metrics["mean_abs_steering"], 0.3)
        self.assertAlmostEqual(metrics["mean_throttle_command"], 0.25)
        self.assertAlmostEqual(metrics["mean_brake_command"], 0.125)
        self.assertAlmostEqual(metrics["mean_abs_longitudinal_delta"], 0.75)

    def test_summary_preserves_outcome_rates(self):
        report = summarise_episode_rows([
            {"outcome": "success", "steps": 10, "mean_speed_mps": 5.0},
            {"outcome": "off_road", "steps": 20, "mean_speed_mps": 7.0},
        ])
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["success_rate"], 0.5)
        self.assertEqual(report["off_road_rate"], 0.5)
        self.assertEqual(report["mean_steps"], 15.0)


if __name__ == "__main__":
    unittest.main()
