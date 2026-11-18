import unittest
from types import SimpleNamespace

import numpy as np

from bev_policy.evaluate import build_perception_dashboard


class TestBEVEvaluationView(unittest.TestCase):
    def test_dashboard_shows_five_cameras_and_front_stereo_depth(self):
        colours = {
            "front_left_camera": (10, 20, 30),
            "front_right_camera": (40, 50, 60),
            "left_camera": (70, 80, 90),
            "right_camera": (100, 110, 120),
            "rear_camera": (130, 140, 150),
        }
        frames = {
            name: np.full((30, 40, 3), colour, dtype=np.uint8)
            for name, colour in colours.items()
        }
        detections = {
            name: [] for name in (
                "front_left_camera", "left_camera", "right_camera", "rear_camera"
            )
        }
        state = SimpleNamespace(
            bev_grid=np.zeros((64, 64), dtype=np.float32),
            scalar_state=np.asarray((5.0, 0.1, 0.0, 0.0, 0.0, 100.0), dtype=np.float32),
        )
        simulation_view = np.full((30, 40, 3), (200, 190, 180), dtype=np.uint8)
        dashboard = build_perception_dashboard(
            simulation_view, frames, detections,
            np.zeros((30, 40), dtype=np.float32), state, np.asarray((0.1, 0.2)),
            "fog", 0.8, 12, "BC",
        )

        self.assertEqual(dashboard.shape, (872, 1080, 3))
        np.testing.assert_array_equal(dashboard[180, 270], (200, 190, 180))
        np.testing.assert_array_equal(dashboard[470, 180], colours["front_left_camera"])
        np.testing.assert_array_equal(dashboard[470, 540], colours["front_right_camera"])
        np.testing.assert_array_equal(dashboard[470, 900], (0, 0, 0))
        np.testing.assert_array_equal(dashboard[740, 180], colours["left_camera"])
        np.testing.assert_array_equal(dashboard[740, 540], colours["right_camera"])
        np.testing.assert_array_equal(dashboard[740, 900], colours["rear_camera"])


if __name__ == "__main__":
    unittest.main()
