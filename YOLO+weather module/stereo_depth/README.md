# Stereo depth utilities

This directory provides reusable perception utilities used by the MetaDrive
camera prototype. It is a component library, not a deployed system and not an
integrated RL environment.

## What the code provides

- StereoSGBM disparity and front depth conversion:
  `depth_m = focal_length_px × baseline_m / disparity_px`.
- Camera and BEV geometry helpers.
- Front YOLO-box depth association.
- Side/rear ground-plane position estimates.
- A standalone 64×64 occupancy-grid utility.

The configured camera assumptions are 1200×900 pixels, 60° horizontal FOV,
1000 px focal length, and 0.5 m front-stereo baseline. The 64×64 grid covers
approximately ±10 m laterally, +17.5 m ahead, and −2.5 m behind the ego
vehicle.

## Run checks

From `YOLO+weather module`:

```powershell
python .\stereo_depth\test_module.py
```

The checks cover module imports, calibration constants, coordinate transforms,
disparity/depth arithmetic, and synthetic grid construction. They do not use
recorded camera data or MetaDrive depth ground truth.

## Required validation before research claims

1. Capture synchronized front-left/right frames using
   `camera/multi_camera_drive.py`.
2. Compare estimated depth with simulator ground truth for known objects.
3. Check pixel-to-ego and ego-to-grid placement visually and quantitatively.
4. Validate the final BEV tensor contract once it is integrated with route,
   lane-boundary, scalar-state, and policy code.

Do not quote theoretical depth-error figures or module test passes as measured
system accuracy.
