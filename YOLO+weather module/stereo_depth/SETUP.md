# Stereo Depth Estimation - Setup & Testing Guide

Complete setup and usage guide for the stereo depth estimation module.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Troubleshooting](#troubleshooting)
4. [Examples](#examples)
5. [Integration with YOLO](#integration-with-yolo)
6. [Performance Optimization](#performance-optimization)

---

## Installation

### Step 1: Install Required Packages

Make sure your Python virtual environment is activated:

```powershell
# PowerShell
cd e:\Capstone\Minimal_Grid_env
.\.venv\Scripts\Activate.ps1

# Or Linux/Mac
source venv/bin/activate
```

Install the requirements:

```bash
cd "e:\Capstone\Minimal_Grid_env\YOLO+weather module"
pip install -r requirements.txt
```

### Step 2: Verify Installation

Test that all imports work:

```bash
python -c "from stereo_depth import DepthProcessor; print('✓ Import successful')"
```

Expected output:
```
✓ Import successful
```

### Step 3: Optional - Install opencv-contrib for Better Results

For best performance with Weighted Least Squares filtering:

```bash
pip install opencv-contrib-python
```

Note: If `opencv-contrib-python` is not installed, the system will automatically fall back to standard filtering. You'll see:

```
⚠ WLS filter disabled (opencv-contrib-python not available)
  To enable: pip install opencv-contrib-python
```

---

## Quick Start

### Example 1: Basic Stereo Depth Estimation

Create a file `test_stereo.py`:

```python
from stereo_depth import DepthProcessor
import cv2

# Initialize processor
processor = DepthProcessor()

# Load images
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Compute depth
result = processor.process_stereo_pair(left_img, right_img)

depth_map = result['depth']
print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f} m")

# Visualize
depth_vis = processor.depth_to_visualization(depth_map)
cv2.imshow('Depth Map', depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Run it:

```bash
python test_stereo.py
```

### Example 2: Using the Visualization Script

If you have stereo images:

```powershell
python "e:\Capstone\Minimal_Grid_env\YOLO+weather module\stereo_depth\visualize_stereo.py" `
    "C:\path\to\left.jpg" "C:\path\to\right.jpg"
```

**Controls in visualization window:**
- `d` - Show depth map
- `s` - Show disparity map
- `l` - Show left image
- `r` - Show right image
- `c` - Print calibration summary
- `q` - Quit

Results are automatically saved to `stereo_results/` folder.

### Example 3: Print Camera Calibration

```python
from stereo_depth import CameraParameters

camera = CameraParameters()
camera.print_summary()
```

Expected output:
```
============================================================
CAMERA PARAMETERS SUMMARY
============================================================

INTRINSIC PARAMETERS:
  Image Resolution: 1200 x 900 px
  Focal Length: 1000 px
  Principal Point: (600.0, 450.0) px
  FOV (horizontal): 60°
  Camera Height: 1.4 m
  Downward Pitch: -5°

STEREO PARAMETERS:
  Baseline: 0.5 m
  Disparity Range: 1-192 pixels

DEPTH PRECISION:
  @ 10m: ±0.04m (0.4%)
  @ 20m: ±0.16m (0.8%)
  @ 30m: ±0.36m (1.2%)

CAMERA CONFIGURATIONS:
  FRONT: offset=(2.0m, 1.4m), yaw=0°, pitch=-5°, role=stereo_left_primary
  LEFT: offset=(0.0m, 1.4m), yaw=-90°, pitch=-5°, role=monocular_side
  RIGHT: offset=(0.0m, 1.4m), yaw=90°, pitch=-5°, role=monocular_side
  REAR: offset=(-2.0m, 1.4m), yaw=180°, pitch=-5°, role=monocular_rear
============================================================
```

---

## Troubleshooting

### Error: `AttributeError: module 'cv2' has no attribute 'ximgproc'`

**Cause:** `opencv-contrib-python` not installed or OpenCV version mismatch.

**Solution:** Install opencv-contrib-python:

```bash
pip install opencv-contrib-python
```

Or upgrade both packages:

```bash
pip install --upgrade opencv-python opencv-contrib-python
```

**Note:** The stereo matcher will still work without this - it will just disable WLS filtering.

---

### Error: `ImportError: cannot import name 'DepthProcessor'`

**Cause:** Python path not configured correctly or package not in PYTHONPATH.

**Solution:** 

Ensure you're running from the correct directory:

```bash
cd "e:\Capstone\Minimal_Grid_env\YOLO+weather module"
python your_script.py
```

Or explicitly add to PYTHONPATH:

```bash
$env:PYTHONPATH = "e:\Capstone\Minimal_Grid_env\YOLO+weather module;$env:PYTHONPATH"
python your_script.py
```

---

### Error: `module 'cv2' has no attribute 'StereoSGBM_create'`

**Cause:** Very old OpenCV version.

**Solution:** Update OpenCV:

```bash
pip install --upgrade opencv-python
```

Minimum version required: **4.6.0**

---

### Slow Performance / High Memory Usage

**Causes:**
- Large input images (> 1200×900)
- Running on CPU instead of GPU
- Not using optimized disparity parameters

**Solutions:**

1. **Resize images before processing:**

```python
import cv2

left_img = cv2.imread('left.jpg')
left_img_small = cv2.resize(left_img, (600, 450))  # Half size
```

2. **Use faster but less accurate parameters:**

```python
from stereo_depth import StereoMatcher
import cv2

matcher = StereoMatcher()

# For faster (but noisier) results
matcher.matcher.setNumDisparities(96)  # Instead of 192
matcher.matcher.setBlockSize(3)        # Instead of 5
```

3. **Disable point cloud computation:**

```python
# Skip expensive point cloud generation
result = processor.process_stereo_pair(left_img, right_img, compute_point_cloud=False)
```

---

## Examples

### Example 1: Full Pipeline - Stereo + YOLO + BEV

```python
from stereo_depth import DepthProcessor, CameraPosition
import cv2

# Initialize
processor = DepthProcessor()

# Load stereo images
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Compute depth
result = processor.process_stereo_pair(left_img, right_img)
depth_map = result['depth']

# Simulate YOLO detections
detections = [
    {
        'bbox': [100, 150, 250, 400],
        'confidence': 0.92,
        'class_id': 2,  # car
        'class_name': 'car'
    },
    {
        'bbox': [500, 200, 700, 500],
        'confidence': 0.88,
        'class_id': 7,  # truck
        'class_name': 'truck'
    }
]

# Process detections with depth
detections_by_camera = {CameraPosition.FRONT: detections}
depth_maps = {CameraPosition.FRONT: depth_map}

bev_grid = processor.process_detections_to_bev(detections_by_camera, depth_maps)

# Visualize
bev_vis = processor.bev_to_visualization()
cv2.imshow('BEV Grid', bev_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Example 2: Coordinate Transformations

```python
from stereo_depth import CoordinateTransform, CameraParameters

transform = CoordinateTransform(CameraParameters())

# Test pixel -> 3D -> ego -> BEV
u, v = 600, 450  # Center of image
depth = 20.0     # 20 meters away

# Step-by-step transformation
X_cam, Y_cam, Z_cam = transform.pixel_to_camera_3d(u, v, depth)
print(f"Camera 3D: X={X_cam:.2f}, Y={Y_cam:.2f}, Z={Z_cam:.2f}")

X_ego, Z_ego = transform.camera_to_ego_frame(X_cam, Z_cam, yaw_deg=0)
print(f"Ego frame: X={X_ego:.2f}, Z={Z_ego:.2f}")

col, row = transform.ego_to_bev_grid(X_ego, Z_ego)
print(f"BEV grid: col={col:.1f}, row={row:.1f}")
```

### Example 3: Depth Precision Analysis

```python
from stereo_depth import DisparityToDepth, CameraParameters

converter = DisparityToDepth(CameraParameters())

distances = [5, 10, 15, 20, 25, 30]
for dist in distances:
    error = converter.depth_error_at_distance(dist)
    rel_error = (error / dist) * 100
    print(f"{dist}m: ±{error:.3f}m ({rel_error:.1f}%)")
```

Expected output:
```
5m: ±0.010m (0.2%)
10m: ±0.040m (0.4%)
15m: ±0.090m (0.6%)
20m: ±0.160m (0.8%)
25m: ±0.250m (1.0%)
30m: ±0.360m (1.2%)
```

---

## Integration with YOLO

### Full Integration Example

```python
from ultralytics import YOLO
from stereo_depth import DepthProcessor, CameraPosition
import cv2

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Initialize depth processor
processor = DepthProcessor()

# Load stereo images
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Step 1: Compute stereo depth
result = processor.process_stereo_pair(left_img, right_img)
depth_map = result['depth']

# Step 2: Run YOLO detection
yolo_results = yolo_model(left_img, conf=0.4)[0]

# Step 3: Convert YOLO results to detection format
detections = []
for box in yolo_results.boxes:
    class_id = int(box.cls.item())
    
    # Filter to obstacle classes (cars, trucks, buses, etc.)
    if class_id not in {0, 2, 3, 5, 7}:
        continue
    
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    confidence = float(box.conf.item())
    
    detections.append({
        'bbox': [x1, y1, x2, y2],
        'confidence': confidence,
        'class_id': class_id,
        'class_name': yolo_results.names.get(class_id)
    })

# Step 4: Enhance detections with depth and build BEV
detections_by_camera = {CameraPosition.FRONT: detections}
depth_maps = {CameraPosition.FRONT: depth_map}

bev_grid = processor.process_detections_to_bev(detections_by_camera, depth_maps)

# Step 5: Visualize and save
bev_vis = processor.bev_to_visualization()
cv2.imwrite('bev_grid.jpg', bev_vis)

print(f"Found {len(detections)} obstacles")
print(f"BEV grid saved to bev_grid.jpg")
```

---

## Performance Optimization

### Batch Processing

For processing many stereo pairs:

```python
from stereo_depth import DepthProcessor
import cv2
import glob

processor = DepthProcessor()
image_pairs = glob.glob("stereo_pairs/left_*.jpg")

for left_path in sorted(image_pairs):
    right_path = left_path.replace('left_', 'right_')
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    # Process without point cloud for speed
    result = processor.process_stereo_pair(
        left_img, right_img, 
        compute_point_cloud=False  # Save time
    )
    
    print(f"Processed {left_path}")
```

### GPU Acceleration

The stereo matching algorithm runs on CPU. For GPU acceleration, consider:

1. **CUDA-enabled OpenCV** (requires compilation)
2. **Tegra Accelerated Stereo** (NVIDIA hardware)
3. **ONNX Runtime** with GPU backend

---

## Tips & Best Practices

✅ **DO:**
- Use rectified stereo images for best results
- Ensure stereo images are properly synchronized
- Use consistent lighting during capture
- Test calibration parameters before large-scale processing

❌ **DON'T:**
- Mix different camera models without recalibration
- Process heavily corrupted or occluded regions
- Expect accurate depth in textureless areas
- Use stereo matching on single images

---

## References

- Technical Specification: Vision-Based BEV Perception for Energy-Optimal Autonomous Driving using PPO
- OpenCV Stereo Documentation: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html
- Semi-Global Matching (SGM): Hirschmüller, H. "Stereo Processing by Semiglobal Matching and Mutual Information" (2008)

