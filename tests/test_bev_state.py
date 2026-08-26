import unittest
from types import SimpleNamespace

import numpy as np

from bev_state import BEVStateAssembler, BEVStateConfig


class FakeLane:
    length = 50.0

    def local_coordinates(self, position):
        return 10.0, 0.75

    def heading_theta_at(self, longitudinal):
        return 0.10 + 0.01 * longitudinal

    def position(self, longitudinal, lateral):
        return np.array([longitudinal, lateral], dtype=np.float32)


class TestBEVStateAssembler(unittest.TestCase):
    def test_output_contract_with_direct_detections(self):
        assembler = BEVStateAssembler()
        obs = assembler.assemble(
            detections_by_camera={
                "right_camera": [
                    {
                        "xmin": 550,
                        "ymin": 500,
                        "xmax": 650,
                        "ymax": 700,
                        "confidence": 0.9,
                        "label": "car",
                    }
                ]
            },
            info={"velocity": np.array([3.0, 4.0]), "route_completion": 0.4, "distance_to_goal": 42.0},
        )

        self.assertEqual(obs.bev_grid.shape, (64, 64))
        self.assertEqual(obs.bev_grid.dtype, np.float32)
        self.assertEqual(obs.scalar_state.shape, (6,))
        self.assertEqual(obs.scalar_state.dtype, np.float32)
        self.assertAlmostEqual(float(obs.scalar_state[0]), 5.0)
        self.assertAlmostEqual(float(obs.scalar_state[1]), 0.4)
        self.assertAlmostEqual(float(obs.scalar_state[5]), 42.0)
        self.assertEqual(len(obs.objects), 1)
        self.assertTrue(np.any(obs.bev_grid == assembler.config.values.occupied))
        self.assertTrue(np.any(obs.bev_grid == assembler.config.values.unknown))
        self.assertTrue(np.any(obs.bev_grid == assembler.config.values.free))

    def test_physical_footprint_scales_to_grid_cells(self):
        assembler = BEVStateAssembler()
        grid = assembler.build_bev_grid([
            assembler.objects_from_detections(
                {"right_camera": [{"xmin": 580, "ymin": 500, "xmax": 620, "ymax": 700, "confidence": 1.0, "label": "car"}]},
                None,
            )[0]
        ])
        occupied = int(np.sum(grid == assembler.config.values.occupied))
        self.assertGreaterEqual(occupied, 40)

    def test_ego_state_uses_lane_when_available(self):
        assembler = BEVStateAssembler()
        env = SimpleNamespace(
            agent=SimpleNamespace(
                lane=FakeLane(),
                position=(0.0, 0.0),
                heading_theta=0.25,
                speed=7.0,
                navigation=SimpleNamespace(final_lane=FakeLane(), route_completion=0.25),
            )
        )
        state = assembler.extract_scalar_state(env, {})
        self.assertEqual(state.shape, (6,))
        self.assertAlmostEqual(float(state[0]), 7.0)
        self.assertAlmostEqual(float(state[1]), 0.25)
        self.assertAlmostEqual(float(state[2]), 0.75)
        self.assertAlmostEqual(float(state[3]), 0.05, places=5)
        self.assertAlmostEqual(float(state[4]), 0.01, places=5)
        self.assertAlmostEqual(float(state[5]), 50.0)

    def test_out_of_range_object_is_ignored(self):
        assembler = BEVStateAssembler(BEVStateConfig(camera_range_m=30.0))
        outside = assembler.ego_to_grid(100.0, 100.0)
        self.assertIsNone(outside)


if __name__ == "__main__":
    unittest.main()
