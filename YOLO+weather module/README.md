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

To apply synthetic image corruption to both display and perception:

```powershell
python .\manual_drive_stereo_yolo_weather.py --device cpu --weather fog --level 0.5 --perception-weather
```

Without `--perception-weather`, weather affects only the display. Neither
mode is a physical fog/rain or sensor-noise simulation.

## Scope and limitations

- Front depth is a raw StereoSGBM estimate. It has not been benchmarked against
  MetaDrive ground truth.
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
