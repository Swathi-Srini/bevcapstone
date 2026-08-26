# MetaDrive perception prototype

This folder contains independent perception demonstrations for the project.
It is not an end-to-end BEV controller and does not train a driving policy.

## Current live entry point

From this folder, with the repository virtual environment activated:

```powershell
python .\manual_drive_stereo_yolo_weather.py --device cpu --weather none
```

The program mounts five physical MetaDrive RGB cameras:

- front-left and front-right: parallel front stereo pair;
- left, right and rear: monocular views.

It runs StereoSGBM on the front pair, YOLO on the front/left/right/rear logical
views, overlays the results, and writes detection rows to
`integrated_output/detection_log.csv` on exit. Use `W/A/S/D` to drive and
`Q` to quit.

### Camera calibration note

The technical specification lists a 1200 px-wide image, 60° horizontal FOV,
and `f ≈ 1000 px`. The active runner keeps the specified 60° lens and derives
the exact compatible focal length, `f = 600 / tan(30°) = 1039.23 px`, for
metric depth. The nominal 1000 px figure is retained in code for traceability,
but using it with a true 60° lens produces a systematic depth-scale bias.

To apply synthetic image corruption to both display and perception:

```powershell
python .\manual_drive_stereo_yolo_weather.py --device cpu --weather fog --level 0.5 --perception-weather
```

Without `--perception-weather`, weather affects only the display. When the
flag is enabled, the front-left and front-right frames receive the same
synthetic weather realization so SGBM retains stereo correspondence. Neither
mode is a physical fog/rain or sensor-noise simulation.

## Scope and limitations

- Front depth is a raw StereoSGBM estimate. Run the reproducible MetaDrive
  pose-reference benchmark before making an accuracy claim:

  ```powershell
  python .\ground_truth_stereo_benchmark.py --steps 60 --traffic-density 0.5 --start-seed 0
  ```

  It saves `integrated_output/ground_truth_benchmark/stereo_vs_metadrive_pose_reference.csv`
  and reports error against each visible traffic vehicle's camera-coordinate
  visible surface. This is a simulator pose reference, not a per-pixel
  depth-sensor label; targets occluded by a nearer vehicle are excluded.
- Side/rear positions use a ground-plane estimate, not stereo.
- This script logs per-detection positions; it does not construct the final
  policy BEV or produce a control action.
- YOLO may download `yolov8n.pt` on first use. Run with `--device cpu`
  unless CUDA is deliberately configured.

## Supporting code

- `stereo_depth/`: reusable camera parameters, depth conversion, coordinate
  transforms, and a standalone 64×64 occupancy-grid utility.
- `weather/`: synthetic fog and rain image augmentation.
- `yolo/`: Ultralytics YOLO loading and traffic-class parsing.

Run the lightweight stereo module checks with:

```powershell
python .\stereo_depth\test_module.py
```

These are module-level checks using constants and synthetic inputs; passing
them is not evidence of live depth accuracy or final-system performance.

## Legacy scripts

`manual_drive_with_depth.py`, `realtime_depth_logger.py`, and
`yolo_depth_realtime_logger.py` are older monocular/bounding-box-depth demos.
They are not the current stereo implementation and must not be used to claim
StereoSGBM depth. Use `manual_drive_stereo_yolo_weather.py` for the current
four-direction YOLO + weather + front-stereo workflow.
