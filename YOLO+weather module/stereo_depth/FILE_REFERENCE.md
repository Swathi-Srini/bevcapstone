# Stereo Depth Module - File Structure & Reference

Complete file organization and descriptions for the stereo depth estimation module.

## 📂 Directory Structure

```
stereo_depth/
├── Core Implementation
│   ├── __init__.py                 # Public API exports
│   ├── camera_params.py            # Camera calibration & intrinsics
│   ├── stereo_matcher.py           # SGM stereo matching engine
│   ├── depth_utils.py              # Coordinate transforms & utilities
│   └── depth_processor.py          # High-level pipeline
│
├── Tools & Utilities
│   ├── visualize_stereo.py         # Interactive visualization tool
│   ├── examples.py                 # 4 complete working examples
│   └── test_module.py              # Comprehensive test suite
│
└── Documentation
    ├── README.md                   # Full module documentation
    ├── SETUP.md                    # Installation & troubleshooting
    ├── INTEGRATION.md              # YOLO & RL integration guide
    ├── SUMMARY.md                  # Implementation overview
    ├── DEPLOYMENT.md               # Deployment & scaling
    └── FILE_REFERENCE.md           # This file
```

---

## 📄 File Descriptions

### Core Implementation Files

#### `__init__.py`
**Purpose**: Public API exports and module initialization

**Exports**:
- `CameraParameters` - Camera calibration
- `CameraPosition` - Camera enum
- `CameraConfig` - Camera configuration
- `StereoMatcher` - Stereo matching engine
- `DepthProcessor` - Main pipeline
- `DisparityToDepth` - Disparity converter
- `CoordinateTransform` - Coordinate transformations
- `PhysicalSizeEstimator` - Size estimation
- `MonocularDepth` - Monocular depth

**Key Functions**:
- None (re-exports only)

**Lines**: ~15
**Status**: ✓ Complete

---

#### `camera_params.py`
**Purpose**: Camera calibration and intrinsic parameters

**Main Classes**:
- `CameraPosition(Enum)` - Camera mounting positions (FRONT, LEFT, RIGHT, REAR)
- `CameraConfig` - Configuration for single camera mounting
- `CameraParameters` - Main calibration class

**Key Features**:
- Intrinsic matrix K computation
- Stereo baseline configuration (0.5m)
- Depth precision analysis
- All 4 camera configurations
- FOV to focal length conversion

**Constants** (from tech spec):
- `IMAGE_WIDTH = 1200 px`
- `IMAGE_HEIGHT = 900 px`
- `FOV_HORIZONTAL = 60°`
- `FOCAL_LENGTH = 1000 px`
- `STEREO_BASELINE = 0.5 m`
- `NUM_DISPARITIES = 192`
- `BLOCK_SIZE = 5`

**Key Methods**:
- `get_depth_precision(distance)` - Error @ distance
- `get_camera_config(position)` - Get camera config
- `get_stereo_pair_cameras()` - Get stereo pair
- `print_summary()` - Print calibration info

**Lines**: ~300
**Status**: ✓ Complete

---

#### `stereo_matcher.py`
**Purpose**: OpenCV StereoSGBM stereo matching implementation

**Main Class**:
- `StereoMatcher` - SGM stereo matching engine

**Key Features**:
- Semi-Global Matching algorithm
- Disparity to depth conversion
- WLS filtering (optional)
- Point cloud generation
- Visualization methods

**Key Methods**:
- `compute_disparity(left, right)` - Get disparity map
- `compute_depth(left, right)` - Get depth map
- `disparity_to_depth(disparity)` - Convert disparity
- `get_point_cloud(left, right)` - Generate 3D points
- `get_disparity_visualization()` - Visualize disparity
- `get_depth_visualization()` - Visualize depth

**Error Handling**:
- Graceful fallback if opencv-contrib not available
- WLS filter checked at initialization
- All NaN/inf values handled

**Lines**: ~350
**Status**: ✓ Complete

---

#### `depth_utils.py`
**Purpose**: Utility functions and coordinate transformations

**Main Classes**:
1. `DisparityToDepth` - Disparity ↔ depth conversion
2. `CoordinateTransform` - Pixel ↔ 3D ↔ Ego ↔ BEV transforms
3. `PhysicalSizeEstimator` - Object size estimation
4. `MonocularDepth` - Ground-plane depth estimation

**Key Methods**:

**DisparityToDepth**:
- `disparity_to_depth(d_px)` - Z = f*B / d
- `depth_to_disparity(Z)` - d = f*B / Z
- `depth_error_at_distance(Z)` - Error estimation

**CoordinateTransform**:
- `pixel_to_camera_3d(u, v, Z)` - Back-project
- `camera_to_ego_frame(X_cam, Z_cam, yaw)` - Rotate
- `ego_to_bev_grid(X_ego, Z_ego)` - Project to grid
- `camera_to_bev_grid()` - Full pipeline

**PhysicalSizeEstimator**:
- `estimate_physical_size(w_bbox, h_bbox, Z)` - Size in metres
- `draw_physical_size_on_grid()` - Rectangle mask

**MonocularDepth**:
- `estimate_depth_from_ground_plane(v, pitch, height)` - Ground projection
- `estimate_lateral_offset(u, Z)` - Lateral position

**Lines**: ~400
**Status**: ✓ Complete

---

#### `depth_processor.py`
**Purpose**: High-level perception pipeline

**Main Class**:
- `DepthProcessor` - Coordinates all components

**Key Features**:
- Stereo depth computation
- YOLO detection processing
- Multi-camera detection fusion
- BEV grid construction
- Visualization methods

**Key Methods**:
- `process_stereo_pair(left, right)` - Main stereo processing
- `process_yolo_detection(detection, depth, camera)` - Enhance detection
- `process_detections_to_bev(detections, depths)` - Build BEV grid
- `depth_to_visualization(depth)` - Visualize depth
- `disparity_to_visualization(disparity)` - Visualize disparity
- `bev_to_visualization()` - Visualize BEV
- `print_calibration_summary()` - Print full specs

**Lines**: ~400
**Status**: ✓ Complete

---

### Tools & Utilities Files

#### `visualize_stereo.py`
**Purpose**: Interactive stereo depth visualization tool

**Main Class**:
- `StereoVisualizer` - Interactive viewer

**Usage**:
```bash
python visualize_stereo.py left.jpg right.jpg
```

**Key Methods**:
- `load_images(left, right)` - Load stereo pair
- `compute_depth_maps()` - Compute depth
- `visualize_interactive()` - Show interactive viewer
- `save_results(output_dir)` - Save all outputs

**Controls**:
- `d` - Depth map
- `s` - Disparity map
- `l` - Left image
- `r` - Right image
- `c` - Calibration summary
- `q` - Quit

**Output Files**:
- `depth_map.png` - Depth visualization
- `disparity_map.png` - Disparity visualization
- `left_image.png`, `right_image.png` - Input images
- `depth_map.npy`, `disparity_map.npy` - Raw data
- `point_cloud.npy` - 3D points
- `metadata.txt` - Statistics

**Lines**: ~300
**Status**: ✓ Complete

---

#### `examples.py`
**Purpose**: Complete working examples for all features

**Functions**:
1. `example_depth_estimation()` - Basic stereo depth
2. `example_yolo_integration()` - YOLO + depth
3. `example_bev_grid_construction()` - BEV from detections
4. `example_coordinate_transformations()` - Full pipeline

**Classes**:
- `BEVPerceptionPipeline` - End-to-end system

**Usage**:
```bash
python examples.py
```

**Lines**: ~350
**Status**: ✓ Complete

---

#### `test_module.py`
**Purpose**: Comprehensive module verification suite

**Test Functions**:
1. `test_imports()` - All imports work
2. `test_camera_params()` - Calibration correct
3. `test_coordinate_transforms()` - Transformations valid
4. `test_disparity_conversion()` - D ↔ Z conversion
5. `test_stereo_matcher_initialization()` - SGM setup
6. `test_depth_processor()` - Pipeline works
7. `test_bev_grid()` - BEV construction

**Usage**:
```bash
python test_module.py
```

**Expected Output**: All 7 tests pass ✓

**Lines**: ~350
**Status**: ✓ Complete

---

### Documentation Files

#### `README.md`
**Purpose**: Complete module documentation and reference

**Sections**:
- Features overview
- Installation instructions
- Quick start examples
- Module structure
- Key classes reference
- BEV grid specification
- Camera configuration
- Technical reference
- Performance notes
- Limitations
- Examples

**Audience**: End users, developers integrating the module

**Status**: ✓ Complete

---

#### `SETUP.md`
**Purpose**: Installation, troubleshooting, and quick start

**Sections**:
- Installation steps (Python 3.8+)
- Quick start examples (3 examples)
- Troubleshooting guide (5 common issues)
- Examples (3 detailed examples)
- Integration with YOLO
- Performance optimization
- Tips and best practices
- References

**Audience**: First-time users, troubleshooting

**Status**: ✓ Complete

---

#### `INTEGRATION.md`
**Purpose**: Integration guide for YOLO and RL training

**Sections**:
- Architecture overview (ASCII diagram)
- Step-by-step integration guide
- YOLO depth integration module
- Environment integration
- Example training loop
- Performance breakdown
- Optimization strategies
- Testing integration

**Code Includes**:
- `YOLODepthIntegration` class
- `RealTimePerception` class
- Example RL training code
- Performance analysis

**Audience**: Integrating with existing systems

**Status**: ✓ Complete

---

#### `SUMMARY.md`
**Purpose**: High-level implementation overview

**Sections**:
- What's been implemented
- Module structure
- Key features
- Quick start
- Technical specifications
- Module components
- Performance metrics
- Integration ready
- Testing checklist
- Error handling
- Examples included
- Limitations
- Next steps

**Audience**: Project stakeholders, reviewers

**Status**: ✓ Complete

---

#### `DEPLOYMENT.md`
**Purpose**: Deployment, scaling, and validation

**Sections**:
- Implementation checklist
- Post-installation steps
- Installation instructions (Windows/Linux)
- Usage patterns (3 examples)
- Testing checklist
- Quality metrics
- Troubleshooting reference
- Scaling considerations
- Validation checklist
- Learning path
- Support resources
- Version information
- Next steps

**Audience**: Deployment engineers, operators

**Status**: ✓ Complete

---

#### `FILE_REFERENCE.md`
**Purpose**: This file - Complete file structure reference

**Content**: Descriptions of all files, purposes, key methods

**Audience**: Developers maintaining the code

**Status**: ✓ Complete (This file)

---

## 📊 File Statistics

| File | Lines | Type | Status |
|------|-------|------|--------|
| `__init__.py` | 15 | Code | ✓ |
| `camera_params.py` | 300 | Code | ✓ |
| `stereo_matcher.py` | 350 | Code | ✓ |
| `depth_utils.py` | 400 | Code | ✓ |
| `depth_processor.py` | 400 | Code | ✓ |
| `visualize_stereo.py` | 300 | Tool | ✓ |
| `examples.py` | 350 | Tool | ✓ |
| `test_module.py` | 350 | Test | ✓ |
| **Code Total** | **~2,000** | | |
| | | | |
| `README.md` | 350 | Docs | ✓ |
| `SETUP.md` | 300 | Docs | ✓ |
| `INTEGRATION.md` | 300 | Docs | ✓ |
| `SUMMARY.md` | 250 | Docs | ✓ |
| `DEPLOYMENT.md` | 300 | Docs | ✓ |
| `FILE_REFERENCE.md` | 300 | Docs | ✓ |
| **Docs Total** | **~1,800** | | |
| | | | |
| **GRAND TOTAL** | **~3,800** | | |

---

## 🔗 Key Dependencies

```
stereo_depth/
├── camera_params.py
│   └── Imported by: stereo_matcher, depth_processor, depth_utils
│
├── stereo_matcher.py
│   ├── Depends on: camera_params
│   └── Imported by: depth_processor
│
├── depth_utils.py
│   ├── Depends on: camera_params
│   └── Imported by: depth_processor, depth_processor
│
├── depth_processor.py
│   ├── Depends on: all of above
│   └── Used by: visualize_stereo, examples, test_module
│
├── visualize_stereo.py
│   └── Depends on: depth_processor
│
├── examples.py
│   └── Depends on: depth_processor, camera_params
│
└── test_module.py
    └── Depends on: all modules
```

---

## 🎯 Import Patterns

### Pattern 1: Basic Imports
```python
from stereo_depth import DepthProcessor, CameraParameters
```

### Pattern 2: Full Import
```python
from stereo_depth import (
    CameraParameters, CameraPosition,
    StereoMatcher, DepthProcessor,
    DisparityToDepth, CoordinateTransform,
    PhysicalSizeEstimator, MonocularDepth
)
```

### Pattern 3: Direct Imports
```python
from stereo_depth.depth_processor import DepthProcessor
from stereo_depth.camera_params import CameraParameters
from stereo_depth.depth_utils import CoordinateTransform
```

---

## 🔍 Finding Things

### "How do I..."

**...compute stereo depth?**
- File: `stereo_matcher.py`
- Method: `StereoMatcher.compute_depth()`
- Example: `examples.py` - Example 1

**...transform coordinates?**
- File: `depth_utils.py`
- Class: `CoordinateTransform`
- Example: `examples.py` - Example 4

**...build a BEV grid?**
- File: `depth_processor.py`
- Method: `DepthProcessor.process_detections_to_bev()`
- Example: `examples.py` - Example 3

**...integrate with YOLO?**
- File: `depth_processor.py`
- Method: `DepthProcessor.process_yolo_detection()`
- Guide: `INTEGRATION.md`

**...visualize depth maps?**
- File: `visualize_stereo.py`
- Tool: `StereoVisualizer`
- Usage: `python visualize_stereo.py left.jpg right.jpg`

**...verify everything works?**
- File: `test_module.py`
- Usage: `python test_module.py`

---

## 🚀 Quick Reference

### Most Important Files
1. `depth_processor.py` - Main API entry point
2. `camera_params.py` - Calibration reference
3. `README.md` - Complete documentation
4. `examples.py` - Working examples

### For Integration
1. `depth_processor.py` - Core pipeline
2. `INTEGRATION.md` - Integration guide
3. `examples.py` - YOLO example
4. `stereo_depth/yolo_depth_integration.py` (to create)

### For Troubleshooting
1. `test_module.py` - Self-test
2. `SETUP.md` - Troubleshooting section
3. `visualize_stereo.py` - Debug with real images
4. `camera_params.print_summary()` - Verify calibration

---

## 📋 Checklist for Code Review

- [x] All imports resolved
- [x] Type hints present
- [x] Docstrings complete
- [x] Error handling implemented
- [x] Tests provided
- [x] Documentation comprehensive
- [x] Examples working
- [x] No external dependencies except OpenCV, NumPy
- [x] Compatible with Python 3.8+
- [x] PEP 8 compliant

---

## 🎓 Learning Order

1. Start with `README.md` for overview
2. Run `examples.py` to see features
3. Read `SETUP.md` for installation
4. Run `test_module.py` to verify
5. Read relevant sections in code
6. Study `INTEGRATION.md` for your use case
7. Review `DEPLOYMENT.md` before production

---

**Last Updated**: March 2025
**Module Version**: 1.0.0

