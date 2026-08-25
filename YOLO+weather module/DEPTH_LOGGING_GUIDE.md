# Integrated YOLO + Weather + StereoSGBM Logging

Use `manual_drive_stereo_yolo_weather.py` for the supported live integration. Older `manual_drive_with_depth.py` and bounding-box-based logger scripts are retained as historical prototypes; they do not provide the calibrated front stereo pipeline.

## Run

From the repository root:

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather rain --level 0.5 --traffic-density 0.5
```

Controls are `W/A/S/D`; press `Q` to quit. The terminal prints a clean table per simulation step and writes `integrated_output\detection_log.csv`.

## Camera and depth contract

- Five physical RGB camera streams: front-left/front-right stereo pair, left, right, rear.
- YOLO runs on four logical views: front-left (front), left, right, and rear. The front-right stream is the StereoSGBM partner, not a duplicate front detector.
- Front depth uses OpenCV StereoSGBM with `numDisparities=192`, `blockSize=5`, `P1=600`, `P2=2400`, `uniquenessRatio=10`, and SGBM 3-way mode.
- The calibrated depth equation is `Z = fB / d_px = 500 / d_px`, with `f = 1000 px` and `B = 0.5 m`.
- The logged uncertainty is `sigma_Z = Z^2 / 2500` for the specified 0.2-pixel disparity error.

## Weather experiments

Weather is shown in the visualization by default, while YOLO and StereoSGBM use clean paired frames. This avoids false disparity caused by independently generated rain streaks.

To deliberately measure weather-related perception degradation, use:

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather fog --level 0.5 --perception-weather
```

The CSV records whether each row used `clean` or `weathered` perception in the `perception_source` column.

For a repeatable saved-image comparison, run `benchmark_weather_perception.py` from the repository root. It saves clean/weathered annotated images and a detection-count/confidence summary CSV:

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\benchmark_weather_perception.py" --image .\path\to\image.png --weather rain --level 0.5 --device cpu
```

## Verify before handoff

```powershell
Push-Location '.\YOLO+weather module'
..\venv\Scripts\python.exe .\stereo_depth\test_module.py
Pop-Location
```

Expected result: `7/7 tests passed`.
