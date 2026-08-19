# Person 1 — MetaDrive Multi-Camera + Stereo Depth Capture

**Goal:** get RGB frames from all 4 directions around the ego vehicle (front as a stereo pair), drive manually, save frames + a stereo depth map — this is the Sensor Layer of the architecture diagram in the spec PDF.

## 1. Setup

```bash
pip install metadrive-simulator opencv-python numpy
```

No admin rights, no virtualenv needed — MetaDrive's `manual_control=True` reads keys through its own native render window.

## 2. Run it

```bash
python metadrive_stereo_capture.py
```

- Drive with **Arrow Keys or WASD** in the MetaDrive window.
- Every 10 steps it saves a frame from all 5 cameras + a depth map from the front stereo pair.
- Stop with **Ctrl+C** or close the window.

Output structure:

```
captured_frames_stereo/
├── front_left_camera/
├── front_right_camera/
├── left_camera/
├── right_camera/
├── rear_camera/
└── front_depth/
    ├── frame_00000.png   <- colorized depth (for viewing)
    └── frame_00000.npy   <- raw metric depth in meters (for actual use)
```

Frame numbers line up across all folders — `frame_00010` is the same simulation instant everywhere.

## 3. Camera specs used (from the spec PDF, Table 2/3/5)

| Camera | Forward offset | Height | Yaw | Pitch |
|---|---|---|---|---|
| Front-left | +2.0m, -0.25m lateral | 1.4m | 0° | -5° |
| Front-right | +2.0m, +0.25m lateral | 1.4m | 0° | -5° |
| Left | 0.0m | 1.4m | -90° | -5° |
| Right | 0.0m | 1.4m | +90° | -5° |
| Rear | -2.0m | 1.4m | 180° | -5° |

Stereo depth uses `numDisparities=192, blockSize=5, P1=600, P2=2400, disp12MaxDiff=1, uniquenessRatio=10, mode=SGBM_3WAY` — copied verbatim from Table 5, and `Z = (f × B) / disparity` with `f=1000px, B=0.5m` from Sec 3.1–3.2.

**Resolution note:** capturing at 640×480 instead of the doc's 1200×900 for FPS while driving live. If you change `CAM_W`, you may need to recompute `FOCAL_LENGTH_PX` for accurate metric depth (it currently assumes the doc's 1200px-width, 60° FOV setup).

**Unverified:** MetaDrive's `RGBCamera` doesn't have an obvious FOV-override in its basic constructor. If your actual FOV differs from 60°, depth values will be off by a scale factor — sanity-check by driving toward something of known size.

## 4. How the rest of the team uses this

- **BEV person:** reuse the `pos=(x=right, y=forward, z=up)` / `hpr` convention for ego-frame transforms. `front_depth/*.npy` gives you real metric distances to project into the grid without needing YOLO+stereo math redone.
- **YOLO + weather person:** point straight at `front_left_camera/`, `left_camera/`, etc. as a static image dataset — no live MetaDrive needed.
- **BC person:** reuse `build_env()` + `mount_cameras()`, log `(frame, action)` pairs in memory each step instead of writing to disk.

## 5. Known limitations

- Depth is only computed for the front pair — side/rear use monocular projection elsewhere in the pipeline (not this file).
- FOV assumption unverified (see above).
- Frame saving throttled to every 10 steps — lower this if you need denser sequences.
