# Five-camera capture module

This module is the sensor-layer prototype for the MetaDrive project. It mounts
five RGB streams on the ego vehicle:

- front-left and front-right, forming a 0.5 m stereo pair;
- left, right, and rear cameras.

It saves synchronized frames and a front StereoSGBM depth estimate. It does
not run YOLO, construct the final BEV, or train a control policy.

## Run

From the repository root, after activating the virtual environment:

```powershell
python .\camera\multi_camera_drive.py
```

Drive with WASD or arrow keys. Stop with Ctrl+C or by closing the MetaDrive
window.

Useful options:

```powershell
python .\camera\multi_camera_drive.py --steps 2000 --save-every 10
```

## Output

captured_frames_stereo is created under this directory:

```text
front_left_camera/   front_right_camera/   left_camera/
right_camera/        rear_camera/          front_depth/
```

Matching frame numbers refer to the same simulation step. front_depth .npy
files contain raw depth in metres; their PNG counterparts are display
visualisations.

## Validation requirement

Depth uses Z = focal length × baseline / disparity with the configured 1000 px
focal length and 0.5 m baseline. Validate it against MetaDrive ground truth
before using depth accuracy in a report.
