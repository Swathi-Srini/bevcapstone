"""
Stereo Depth Module - Verification & Self-Test

Tests all components of the stereo depth module to ensure proper installation
and functionality.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
_this_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_this_dir, os.pardir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import cv2


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from stereo_depth import (
            CameraParameters,
            StereoMatcher,
            DepthProcessor,
            CameraPosition
        )
        from stereo_depth import (
            DisparityToDepth,
            CoordinateTransform,
            PhysicalSizeEstimator,
            MonocularDepth
        )
        print("[PASS] All imports successful")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_camera_params():
    """Test camera parameters."""
    print("\n" + "="*70)
    print("TEST 2: Camera Parameters")
    print("="*70)
    
    try:
        from stereo_depth import CameraParameters
        
        camera = CameraParameters()
        
        # Check key properties
        assert camera.FOCAL_LENGTH == 1000, "Focal length incorrect"
        assert camera.STEREO_BASELINE == 0.5, "Baseline incorrect"
        assert camera.IMAGE_WIDTH == 1200, "Image width incorrect"
        assert camera.IMAGE_HEIGHT == 900, "Image height incorrect"
        
        # Check depth precision
        precision_30m = camera.get_depth_precision(30)
        assert 0.3 < precision_30m < 0.4, "Depth precision out of range"
        
        print(f"[PASS] Focal length: {camera.FOCAL_LENGTH} px")
        print(f"[PASS] Stereo baseline: {camera.STEREO_BASELINE} m")
        print(f"[PASS] Resolution: {camera.IMAGE_WIDTH}x{camera.IMAGE_HEIGHT} px")
        print(f"[PASS] Depth precision @ 30m: +/-{precision_30m:.3f} m")
        
        return True
    except Exception as e:
        print(f"[FAIL] Camera parameters test failed: {e}")
        return False


def test_coordinate_transforms():
    """Test coordinate transformations."""
    print("\n" + "="*70)
    print("TEST 3: Coordinate Transformations")
    print("="*70)
    
    try:
        from stereo_depth import CameraParameters, CoordinateTransform
        
        camera = CameraParameters()
        transform = CoordinateTransform(camera)
        
        # Test pixel to camera 3D
        u, v = 600, 450  # Center of image
        # The configured BEV forward extent is 17.5 m, so keep the test
        # point inside the declared 64x64 grid rather than testing clipping.
        depth = 15.0
        X_cam, Y_cam, Z_cam = transform.pixel_to_camera_3d(u, v, depth)
        
        assert Z_cam == depth, "Depth not preserved in pixel_to_camera_3d"
        assert abs(X_cam) < 0.1, "X should be near zero (center of image)"
        assert abs(Y_cam) < 0.1, "Y should be near zero (center of image)"
        
        print(f"[PASS] Pixel (600, 450) @ 15m -> Camera 3D: ({X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f})")
        
        # Test camera to ego frame
        X_ego, Z_ego = transform.camera_to_ego_frame(X_cam, Z_cam, camera_yaw_deg=0)
        assert abs(X_ego - X_cam) < 0.01, "Camera to ego transform failed"
        assert abs(Z_ego - Z_cam) < 0.01, "Camera to ego transform failed"
        
        print(f"[PASS] Camera to ego frame (yaw=0 deg): ({X_ego:.3f}, {Z_ego:.3f})")
        
        # Test ego to BEV grid
        col, row = transform.ego_to_bev_grid(X_ego, Z_ego)
        assert 0 <= col <= 64, "BEV column out of range"
        assert 0 <= row <= 64, "BEV row out of range"
        
        print(f"[PASS] Ego to BEV grid: (col={col:.1f}, row={row:.1f})")
        
        return True
    except Exception as e:
        print(f"[FAIL] Coordinate transform test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_disparity_conversion():
    """Test disparity to depth conversion."""
    print("\n" + "="*70)
    print("TEST 4: Disparity-Depth Conversion")
    print("="*70)
    
    try:
        from stereo_depth import DisparityToDepth, CameraParameters
        
        converter = DisparityToDepth(CameraParameters())
        
        # Test disparity to depth
        disparity_px = 25.0
        depth = converter.disparity_to_depth(disparity_px)
        
        expected_depth = 500.0 / disparity_px  # f*B / d = 1000*0.5 / 25
        assert abs(depth - expected_depth) < 0.01, "Disparity-depth conversion failed"
        
        print(f"[PASS] Disparity {disparity_px}px -> Depth {depth:.2f}m")
        
        # Test depth to disparity
        disparity_back = converter.depth_to_disparity(depth)
        assert abs(disparity_back - disparity_px) < 0.1, "Round-trip conversion failed"
        
        print(f"[PASS] Depth {depth:.2f}m -> Disparity {disparity_back:.1f}px")
        
        return True
    except Exception as e:
        print(f"[FAIL] Disparity conversion test failed: {e}")
        return False


def test_stereo_matcher_initialization():
    """Test stereo matcher initialization."""
    print("\n" + "="*70)
    print("TEST 5: Stereo Matcher Initialization")
    print("="*70)
    
    try:
        from stereo_depth import StereoMatcher, CameraParameters
        
        camera = CameraParameters()
        matcher = StereoMatcher(camera)
        
        print("[PASS] Stereo matcher created")
        print(f"  - Algorithm: Semi-Global Matching (SGM)")
        print(f"  - Disparities: {camera.NUM_DISPARITIES}")
        print(f"  - Block size: {camera.BLOCK_SIZE}x{camera.BLOCK_SIZE}")
        
        # Check WLS filter status
        if matcher.use_wls_filter:
            print("  - WLS Filter: Enabled")
        else:
            print(f"  - WLS Filter: Disabled (opencv-contrib-python not installed)")
        
        return True
    except Exception as e:
        print(f"[FAIL] Stereo matcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_depth_processor():
    """Test depth processor."""
    print("\n" + "="*70)
    print("TEST 6: Depth Processor")
    print("="*70)
    
    try:
        from stereo_depth import DepthProcessor, CameraPosition
        import numpy as np
        
        processor = DepthProcessor()
        
        print("[PASS] Depth processor initialized")
        print(f"  - BEV grid size: {processor.bev_grid_size}x{processor.bev_grid_size}")
        print(f"  - Lateral range: +/-{processor.coordinate_transform.bev_lateral_range}m")
        print(f"  - Forward range: +{processor.coordinate_transform.bev_forward_range}m")
        print(f"  - Rear range: -{processor.coordinate_transform.bev_rear_range}m")
        print(f"  - Metres per pixel: {processor.coordinate_transform.metres_per_pixel:.4f}")
        
        # Test with dummy detection
        dummy_detection = {
            'bbox': [100, 150, 250, 300],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }
        
        # Create dummy depth map
        dummy_depth = np.ones((900, 1200), dtype=np.float32) * 15.0
        
        # Process detection
        enhanced = processor.process_yolo_detection(
            dummy_detection, dummy_depth, CameraPosition.FRONT
        )
        
        assert 'ego_position' in enhanced, "Missing ego_position"
        assert 'bev_position' in enhanced, "Missing bev_position"
        assert 'physical_size' in enhanced, "Missing physical_size"
        
        print("[PASS] YOLO detection processing works")
        print(f"  - Ego position: {enhanced['ego_position']}")
        print(f"  - BEV position: {enhanced['bev_position']}")
        print(f"  - Physical size: {enhanced['physical_size']}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Depth processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bev_grid():
    """Test BEV grid construction."""
    print("\n" + "="*70)
    print("TEST 7: BEV Grid Construction")
    print("="*70)
    
    try:
        from stereo_depth import DepthProcessor, CameraPosition
        import numpy as np
        
        processor = DepthProcessor()
        
        # Create multiple dummy detections
        detections = [
            {
                'bbox': [100, 150, 250, 400],
                'confidence': 0.92,
                'class_id': 2,
                'class_name': 'car'
            },
            {
                'bbox': [500, 200, 700, 450],
                'confidence': 0.88,
                'class_id': 7,
                'class_name': 'truck'
            }
        ]
        
        # Create dummy depth map
        depth_map = np.ones((900, 1200), dtype=np.float32) * 20.0
        
        # Build BEV
        detections_by_camera = {CameraPosition.FRONT: detections}
        depth_maps = {CameraPosition.FRONT: depth_map}
        
        bev_grid = processor.process_detections_to_bev(
            detections_by_camera, depth_maps
        )
        
        assert bev_grid.shape == (64, 64), "BEV grid wrong shape"
        assert bev_grid.dtype == np.float32, "BEV grid wrong dtype"
        
        occupied = (bev_grid > 0.5).sum()
        free = (bev_grid < 0.1).sum()
        
        print(f"[PASS] BEV grid constructed: {bev_grid.shape}")
        print(f"  - Occupied pixels: {occupied}")
        print(f"  - Free pixels: {free}")
        print(f"  - Total pixels: {bev_grid.size}")
        print(f"  - Occupancy: {(occupied/bev_grid.size)*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"[FAIL] BEV grid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("STEREO DEPTH MODULE - VERIFICATION TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Camera Parameters", test_camera_params),
        ("Coordinate Transforms", test_coordinate_transforms),
        ("Disparity-Depth Conversion", test_disparity_conversion),
        ("Stereo Matcher Init", test_stereo_matcher_initialization),
        ("Depth Processor", test_depth_processor),
        ("BEV Grid Construction", test_bev_grid),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:8} - {test_name}")
    
    print("-"*70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nALL TESTS PASSED! Module is ready to use.")
        return True
    else:
        print(f"\nWARNING: {total - passed} test(s) failed. Check output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
