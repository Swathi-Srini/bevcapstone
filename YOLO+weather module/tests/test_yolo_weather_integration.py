"""Unit and non-GUI integration tests for the YOLO + weather + stereo module."""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


drive = load_module("manual_drive_stereo_yolo_weather_test", "manual_drive_stereo_yolo_weather.py")
benchmark = load_module("benchmark_weather_perception_test", "benchmark_weather_perception.py")
ground_truth = load_module("ground_truth_stereo_benchmark_test", "ground_truth_stereo_benchmark.py")

from weather.weather_utils import apply_weather, prepare_image
from yolo.yolo_utils import TRAFFIC_CLASS_NAMES, detections_from_results


class FakeStereoMatcher:
    def __init__(self, disparity_px: float):
        self.disparity_px = disparity_px

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.full(left.shape, int(self.disparity_px * 16), dtype=np.int16)


class FakeLens:
    def __init__(self):
        self.fov = None

    def setFov(self, value):
        self.fov = value


class FakeSensor:
    def __init__(self):
        self.lens = FakeLens()
        self.track_args = None

    def track(self, *args):
        self.track_args = args

    def get_lens(self):
        return self.lens


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class TestCameraAndStereoUnit(unittest.TestCase):
    def test_camera_rig_matches_calibration_contract(self):
        self.assertEqual((drive.CAM_W, drive.CAM_H, drive.CAM_FOV), (1200, 900, 60))
        self.assertAlmostEqual(drive.FOCAL_LENGTH_PX, 1039.2304845, places=5)
        self.assertEqual(drive.NOMINAL_FOCAL_LENGTH_PX, 1000.0)
        self.assertEqual(drive.STEREO_BASELINE_M, 0.5)
        self.assertEqual(set(drive.CAMERA_RIGS), {
            "front_left_camera", "front_right_camera", "left_camera", "right_camera", "rear_camera",
        })
        self.assertEqual(drive.YOLO_CAMERAS, ("front_left_camera", "left_camera", "right_camera", "rear_camera"))
        self.assertEqual(drive.CAMERA_RIGS["front_left_camera"][0], (-0.25, 2.0, 1.4))
        self.assertEqual(drive.CAMERA_RIGS["front_right_camera"][0], (0.25, 2.0, 1.4))

    def test_camera_mount_enforces_calibrated_fov_for_every_stream(self):
        sensors = {name: FakeSensor() for name in drive.CAMERA_RIGS}
        environment = SimpleNamespace(
            engine=SimpleNamespace(get_sensor=lambda name: sensors[name]),
            agent=SimpleNamespace(origin=object()),
        )
        drive.mount_camera_rig(environment)
        for name, sensor in sensors.items():
            self.assertEqual(sensor.lens.fov, 60)
            self.assertEqual(sensor.track_args[1:], drive.CAMERA_RIGS[name])

    def test_sgbm_configuration_matches_specification(self):
        matcher = drive.build_stereo_matcher()
        self.assertEqual(matcher.getMinDisparity(), 0)
        self.assertEqual(matcher.getNumDisparities(), 192)
        self.assertEqual(matcher.getBlockSize(), 5)
        self.assertEqual(matcher.getP1(), 600)
        self.assertEqual(matcher.getP2(), 2400)
        self.assertEqual(matcher.getDisp12MaxDiff(), 1)
        self.assertEqual(matcher.getUniquenessRatio(), 10)

    def test_disparity_to_depth_formula_and_range_filtering(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        depth = drive.front_stereo_depth(image, image, FakeStereoMatcher(25.0))
        self.assertTrue(np.allclose(depth, drive.FOCAL_LENGTH_PX * 0.5 / 25.0))
        self.assertTrue(np.all(drive.front_stereo_depth(image, image, FakeStereoMatcher(1000.0)) == 0.0))
        self.assertTrue(np.all(drive.front_stereo_depth(image, image, FakeStereoMatcher(5.0)) == 0.0))

    def test_real_sgbm_recovers_known_synthetic_disparity(self):
        """Exercise OpenCV SGBM itself, not only the depth-conversion wrapper."""
        random = np.random.default_rng(7)
        left = random.integers(0, 256, (180, 400), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-25] = left[:, 25:]  # A known 25-pixel disparity -> 20 m.
        depth = drive.front_stereo_depth(
            cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR),
            drive.build_stereo_matcher(),
        )
        valid = depth[30:-30, 220:-30]
        valid = valid[valid > 0]
        self.assertGreater(valid.size, 1000)
        self.assertAlmostEqual(float(np.median(valid)), drive.FOCAL_LENGTH_PX * 0.5 / 25.0, delta=1.0)

    def test_object_depth_uses_robust_inner_median_and_error_formula(self):
        depth = np.full((100, 100), 20.0, dtype=np.float32)
        depth[20:80, 20:80] = 10.0
        detection = {"xmin": 10, "ymin": 10, "xmax": 90, "ymax": 90}
        self.assertAlmostEqual(drive.object_stereo_depth(detection, depth), 10.0)
        self.assertAlmostEqual(drive.stereo_depth_uncertainty(20.0), 20.0 ** 2 * 0.2 / (drive.FOCAL_LENGTH_PX * 0.5))
        self.assertAlmostEqual(drive.stereo_depth_uncertainty(30.0), 30.0 ** 2 * 0.2 / (drive.FOCAL_LENGTH_PX * 0.5))

    def test_camera_to_ego_projection_for_all_directions(self):
        self.assertEqual(drive.ego_position_from_camera("front_left_camera", 600, 10), (-0.25, 12.0))
        left_x, left_z = drive.ego_position_from_camera("left_camera", 600, 10)
        right_x, right_z = drive.ego_position_from_camera("right_camera", 600, 10)
        self.assertAlmostEqual(left_x, -10.0)
        self.assertAlmostEqual(left_z, 0.0)
        self.assertAlmostEqual(right_x, 10.0)
        self.assertAlmostEqual(right_z, 0.0)
        rear_x, rear_z = drive.ego_position_from_camera("rear_camera", 600, 10)
        self.assertAlmostEqual(rear_x, 0.0)
        self.assertAlmostEqual(rear_z, -12.0)

    def test_front_and_monocular_detection_position_paths(self):
        detection = {"xmin": 550, "ymin": 500, "xmax": 650, "ymax": 700}
        stereo_depth = np.full((900, 1200), 20.0, dtype=np.float32)
        method, z, x, forward = drive.estimate_detection_position("front_left_camera", detection, stereo_depth)
        self.assertEqual(method, "stereo_sgm")
        self.assertAlmostEqual(z, 20.0)
        self.assertIsNotNone(x)
        self.assertIsNotNone(forward)
        method, z, x, forward = drive.estimate_detection_position("right_camera", detection, stereo_depth)
        self.assertEqual(method, "ground_plane_monocular")
        self.assertGreater(z, 1.0)
        self.assertGreater(x, 0.0)
        self.assertIsNotNone(forward)

    def test_ground_truth_benchmark_excludes_occluded_targets(self):
        vehicle = SimpleNamespace(WIDTH=2.0, HEIGHT=1.6)
        target = {"vehicle": vehicle, "forward_m": 20.0, "u": 600.0, "v": 400.0}
        nearer = {"vehicle": vehicle, "forward_m": 10.0, "u": 600.0, "v": 400.0}
        visible = {"vehicle": vehicle, "forward_m": 10.0, "u": 100.0, "v": 100.0}
        self.assertTrue(ground_truth.is_occluded_by_nearer_vehicle(target, [target, nearer], drive.FOCAL_LENGTH_PX))
        self.assertFalse(ground_truth.is_occluded_by_nearer_vehicle(target, [target, visible], drive.FOCAL_LENGTH_PX))


class TestWeatherAndYoloUnit(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 120, 3), 100, dtype=np.uint8)

    def test_weather_preserves_shape_type_and_zero_level_identity(self):
        for weather in ("fog", "rain", "all"):
            output = apply_weather(self.image, weather, 0.0)
            self.assertEqual(output.shape, self.image.shape)
            self.assertEqual(output.dtype, np.uint8)
            self.assertTrue(np.array_equal(output, self.image))

    def test_weather_changes_image_at_nonzero_level(self):
        np.random.seed(42)
        fog = apply_weather(self.image, "fog", 0.5)
        np.random.seed(42)
        rain = apply_weather(self.image, "rain", 0.5)
        self.assertFalse(np.array_equal(fog, self.image))
        self.assertFalse(np.array_equal(rain, self.image))

    def test_stereo_pair_receives_synchronized_weather(self):
        left = np.full((80, 120, 3), 100, dtype=np.uint8)
        right = np.full((80, 120, 3), 150, dtype=np.uint8)
        np.random.seed(123)
        weathered_left, weathered_right = drive.apply_synchronized_stereo_weather(left, right, "rain", 0.5)
        # With the same base image, synchronized weather must be pixel-identical.
        np.random.seed(123)
        same_left, same_right = drive.apply_synchronized_stereo_weather(left, left, "rain", 0.5)
        self.assertTrue(np.array_equal(same_left, same_right))
        self.assertEqual(weathered_left.shape, weathered_right.shape)

    def test_prepare_image_normalizes_supported_sensor_formats(self):
        rgba_float = np.ones((4, 5, 4), dtype=np.float32)
        prepared = prepare_image(rgba_float)
        self.assertEqual(prepared.shape, (4, 5, 3))
        self.assertEqual(prepared.dtype, np.uint8)
        self.assertEqual(prepare_image(np.zeros((4, 5), dtype=np.uint8)).shape, (4, 5, 3))

    def test_yolo_filters_classes_and_confidence(self):
        boxes = SimpleNamespace(
            xyxy=FakeTensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
            conf=FakeTensor([0.90, 0.95, 0.20]), cls=FakeTensor([2, 9, 7]),
        )
        result = SimpleNamespace(boxes=boxes, names={2: "car", 7: "truck", 9: "traffic light"})
        model = SimpleNamespace(model_name="ultralytics_yolo", confidence_threshold=0.4)
        detections = detections_from_results([result], model)
        self.assertEqual(TRAFFIC_CLASS_NAMES, {"person", "car", "motorcycle", "bus", "truck"})
        self.assertEqual(detections, [{"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0,
                                       "confidence": 0.9, "label": "car"}])


class TestNonGuiIntegration(unittest.TestCase):
    def test_visual_annotations_and_depth_tile(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        detections = [{"xmin": 10, "ymin": 20, "xmax": 60, "ymax": 70, "confidence": 0.87, "label": "car"}]
        annotated = drive.annotate_with_depth(image, detections, None)
        self.assertEqual(annotated.shape, image.shape)
        self.assertFalse(np.array_equal(annotated, image))
        self.assertEqual(drive.depth_visualization(np.full((20, 30), 10.0, dtype=np.float32)).shape, (20, 30, 3))

    def test_saved_image_benchmark_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path, output_dir = root / "input.png", root / "output"
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :32], image[:, 32:] = (50, 100, 150), (180, 90, 30)
            cv2.imwrite(str(input_path), image)
            calls = []

            def fake_run_yolo(model, frame):
                calls.append(frame.copy())
                return [{"xmin": 5.0, "ymin": 6.0, "xmax": 30.0, "ymax": 35.0,
                         "confidence": 0.88, "label": "car"}]

            args = ["benchmark_weather_perception.py", "--image", str(input_path), "--weather", "fog",
                    "--level", "0.5", "--output-dir", str(output_dir)]
            with patch.object(benchmark, "ensure_yolo_model", return_value=object()), \
                 patch.object(benchmark, "run_yolo", side_effect=fake_run_yolo), \
                 patch.object(sys, "argv", args):
                self.assertEqual(benchmark.main(), 0)
            self.assertEqual(len(calls), 2)
            self.assertFalse(np.array_equal(calls[0], calls[1]))
            self.assertTrue((output_dir / "clean_annotated.png").is_file())
            self.assertTrue((output_dir / "fog_annotated.png").is_file())
            with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["condition"] for row in rows], ["clean", "weathered"])
            self.assertEqual([row["detections"] for row in rows], ["1", "1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
