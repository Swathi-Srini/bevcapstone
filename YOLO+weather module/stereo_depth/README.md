# Stereo Depth Estimation Module

Comprehensive stereo depth estimation and Bird's Eye View (BEV) perception for autonomous driving using Python, OpenCV, and YOLO object detection.

## Features

✅ **Semi-Global Matching (SGM) Stereo Depth Estimation**
- OpenCV StereoSGBM implementation with optimized parameters
- Disparity to depth conversion (Z = f·B / d_px)
- Depth precision analysis (±0.36m @ 30m)
- Post-processing with Weighted Least Squares filtering

✅ **Multi-Camera Support**
- 4-camera system: Front stereo + Left/Right side + Rear
- Monocular depth via ground-plane projection for side/rear cameras
- Camera-to-ego frame coordinate transformations (yaw-based rotations)
- Calibrated intrinsic parameters (1000px focal length, 60° FOV)

✅ **BEV Grid Construction**
- 64×64 occupancy grid (±10m lateral, +17.5m ahead, -2.5m rear)
- Integration with YOLO detections
- Physical size estimation from bounding boxes
- Per-pixel depth accuracy

✅ **YOLO Integration**
- Multi-camera obstacle detection
- Depth-aware bounding box lifting to 3D
- Physical dimension estimation
- BEV grid population from detections

## Installation

### Prerequisites
- Python 3.8+
- OpenCV with contrib modules (for StereoSGBM and xiproc)
- NumPy

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install opencv-python opencv-contrib-python numpy ultralytics
```

## Quick Start

### Example 1: Basic Stereo Depth

```python
from stereo_depth import DepthProcessor
import cv2

# Initialize processor
processor = DepthProcessor()

# Load stereo images
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Compute depth
result = processor.process_stereo_pair(left_img, right_img)

depth_map = result['depth']          # Depth in metres
point_cloud = result['point_cloud']  # 3D points (N x 3)
```

### Example 2: YOLO + Depth Integration

```python
from stereo_depth import DepthProcessor, CameraPosition

processor = DepthProcessor()

# Process stereo
result = processor.process_stereo_pair(left_img, right_img)
depth_map = result['depth']

# YOLO detection
detection = {
    'bbox': [100, 150, 300, 400],  # [x1, y1, x2, y2]
    'confidence': 0.85,
    'class_id': 2,  # car
    'class_name': 'car'
}

# Enhance with depth
enhanced = processor.process_yolo_detection(
    detection, depth_map, CameraPosition.FRONT
)

print(f"Distance: {enhanced['depth']:.2f}m")
print(f"3D position: {enhanced['ego_position']}")
print(f"Physical size: {enhanced['physical_size']}")
```

### Example 3: Build BEV Grid

```python
from stereo_depth import CameraPosition

# Build BEV from detections
detections_by_camera = {
    CameraPosition.FRONT: [detection1, detection2, ...]
}
depth_maps = {
    CameraPosition.FRONT: depth_map
}

bev_grid = processor.process_detections_to_bev(
    detections_by_camera, depth_maps
)

# Visualize
bev_vis = processor.bev_to_visualization()
cv2.imshow('BEV Grid', bev_vis)
```

## Module Structure

```
stereo_depth/
├── __init__.py              # Public API exports
├── camera_params.py         # Camera calibration and intrinsics
├── stereo_matcher.py        # SGM stereo matching engine
├── depth_utils.py          # Coordinate transforms & utilities
├── depth_processor.py      # High-level pipeline
└── examples.py             # Usage examples and documentation
```

## Key Classes

### `CameraParameters`
Camera calibration with intrinsic matrix, stereo baseline, and depth precision analysis.

```python
camera = CameraParameters()
camera.print_summary()  # Print full calibration

# Access parameters
focal_length = camera.FOCAL_LENGTH  # 1000 px
baseline = camera.STEREO_BASELINE   # 0.5 m
depth_error_at_30m = camera.get_depth_precision(30)  # ±0.36m
```

### `StereoMatcher`
OpenCV StereoSGBM with configurable parameters and post-processing.

```python
matcher = StereoMatcher(camera_params)

# Compute depth
depth = matcher.compute_depth(left_img, right_img)

# Generate point cloud
point_cloud = matcher.get_point_cloud(left_img, right_img)

# Visualizations
depth_vis = matcher.get_depth_visualization(depth)
disp_vis = matcher.get_disparity_visualization(disparity)
```

### `DepthProcessor`
High-level pipeline coordinating all components.

```python
processor = DepthProcessor()

# Process stereo
result = processor.process_stereo_pair(left_img, right_img)

# Integrate YOLO
enhanced_detection = processor.process_yolo_detection(
    detection, depth_map, camera_position
)

# Build BEV
bev_grid = processor.process_detections_to_bev(
    detections_by_camera, depth_maps
)
```

### `CoordinateTransform`
Pixel → 3D Camera → Ego Frame → BEV Grid transformations.

```python
transform = CoordinateTransform(camera_params)

# Pixel to 3D
X, Y, Z = transform.pixel_to_camera_3d(u, v, depth)

# Camera to ego frame
X_ego, Z_ego = transform.camera_to_ego_frame(X, Z, yaw_deg=0)

# Ego to BEV
col, row = transform.ego_to_bev_grid(X_ego, Z_ego)

# One-shot pipeline
col, row = transform.camera_to_bev_grid(u, v, depth, yaw_deg=0)
```

### `PhysicalSizeEstimator`
Estimate object dimensions from bounding box and depth.

```python
estimator = PhysicalSizeEstimator(camera_params)

width_m, length_m = estimator.estimate_physical_size(
    bbox_width_px, bbox_height_px, depth
)
```

### `MonocularDepth`
Ground-plane depth projection for side/rear cameras.

```python
mono = MonocularDepth(camera_params)

# Depth from ground plane
Z = mono.estimate_depth_from_ground_plane(v, camera_pitch_deg, camera_height_m)

# Lateral offset
X = mono.estimate_lateral_offset(u, depth)
```

## BEV Grid Specification

From technical specification Section 5.1:

| Property | Value |
|----------|-------|
| Grid size | 64 × 64 pixels |
| Lateral range | ±10 m (columns 0–63) |
| Forward range | +17.5 m ahead (rows 0–55) |
| Rear range | −2.5 m behind (rows 57–63) |
| Ego position | row = 56, col = 32 |
| Metres per pixel | 0.3125 m/px |

### Grid Values

| Value | Meaning |
|-------|---------|
| 0.0 | Free drivable space |
| 0.5 | Route centreline (HD map) |
| 0.8 | Road boundary (HD map) |
| 1.0 | Physically occupied (ego or obstacle) |

## Camera Configuration

All four cameras share identical intrinsic parameters:

| Parameter | Value |
|-----------|-------|
| Resolution | 1200 × 900 px |
| Focal length | 1000 px |
| FOV (horizontal) | 60° |
| Height | 1.4 m above ground |
| Pitch | −5° (downward) |

### Stereo Baseline
- **B = 0.5 m** (0.25 m each side of vehicle centreline)
- Depth range: 1–30 m
- Depth precision @ 30m: **±0.36 m** (1.2% relative error)

### Stereo Matching Parameters (SGM)

| Parameter | Value |
|-----------|-------|
| Algorithm | Semi-Global Matching (SGM) |
| numDisparities | 192 (divisible by 16) |
| blockSize | 5 |
| P1 | 600 |
| P2 | 2400 |
| disp12MaxDiff | 1 |
| uniquenessRatio | 10 |
| Mode | STEREO_SGBM_MODE_SGBM_3WAY |

## Coordinate Systems

### Camera Frame
- Origin at camera lens center
- Z forward (optical axis), X right, Y down (right-handed)

### Ego Frame
- Origin at vehicle centre
- Z forward (vehicle heading), X right, Y down
- Rotation from camera frame determined by camera yaw

### BEV Grid
- 2D overhead view (looking down)
- Rows = forward/backward (0 = far ahead, 63 = far behind)
- Columns = lateral (0 = far left, 63 = far right)
- Ego vehicle at row 56, col 32

## Technical Reference

This implementation follows the **Internal Technical Specification** for "Vision-Based BEV Perception for Energy-Optimal Autonomous Driving using PPO" (March 11, 2026).

Key sections:
- **Section 2**: Camera Setup
- **Section 3**: Stereo Depth Estimation
- **Section 5**: BEV Grid Construction
- **Section 5.6**: Monocular Depth (side/rear cameras)
- **Section 5.7**: Physical Size Estimation

## Performance Notes

- **Depth accuracy**: ±0.04m @ 10m, ±0.16m @ 20m, ±0.36m @ 30m
- **Computation**: Stereo matching ~50-100ms per frame (GPU recommended)
- **Memory**: Point cloud ~8GB for 900×1200 stereo at 16-bit precision
- **Filtering**: WLS post-processing improves depth at edges (~20% overhead)

## Limitations

1. **Occlusion**: Stereo fails on textureless surfaces and strong occlusions
2. **Baseline constraint**: B = 0.5m limits close-range precision
3. **Monocular fallback**: Side/rear cameras limited to known camera height
4. **Frame rate**: 25 Hz (0.04s timestep) for CARLA synchronous mode

## Examples

See `examples.py` for complete working examples:

```bash
python stereo_depth/examples.py
```

Includes:
1. Basic stereo depth estimation
2. YOLO detection + depth integration
3. BEV grid construction
4. Coordinate transformation pipeline

## Citation

If you use this module in research, please cite the internal technical specification:

```
Vision-Based BEV Perception for Energy-Optimal Autonomous Driving using PPO
Internal Technical Specification
March 11, 2026
```

## License

Internal use only - Capstone Project

## Author

Development based on CARLA Autonomous Driving specification and
Semi-Global Matching algorithm (Hirschmüller et al., 2008).
