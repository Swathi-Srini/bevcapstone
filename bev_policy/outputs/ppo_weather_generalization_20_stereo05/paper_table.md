# PPO weather generalization study

## Experimental setup

BC-initialized safety-shaped PPO checkpoint; 20 paired held-out seeds per condition (30000–30019), MetaDrive traffic density 0.10, 500-step limit, YOLOv8n at 320 px and confidence 0.25, stereo matching at 0.5 scale. The condition uses synthetic camera weather corruption; road friction and vehicle dynamics are unchanged. The BEV builder receives the synthetic visibility horizon associated with the weather condition, so this is not a camera-only weather-estimation evaluation.

## Driving outcomes

Success intervals are 95% Wilson intervals. Continuous values are mean ± SD across episodes.

| Weather | Success | Collision | Off road | Max steps | Speed (m/s) | Lateral error (m) | Heading error (rad) | Estimated energy (Wh/km) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_0.00 | 12/20 (60.0%; 95% CI 38.7%–78.1%) | 8/20 (40.0%) | 0 | 0 | 7.95 ± 0.37 | 0.23 ± 0.07 | 0.017 ± 0.015 | 163.94 ± 69.02 |
| fog_0.25 | 11/20 (55.0%; 95% CI 34.2%–74.2%) | 8/20 (40.0%) | 1 | 0 | 7.98 ± 0.37 | 0.27 ± 0.08 | 0.020 ± 0.018 | 168.89 ± 70.64 |
| fog_0.75 | 12/20 (60.0%; 95% CI 38.7%–78.1%) | 8/20 (40.0%) | 0 | 0 | 8.09 ± 0.38 | 0.31 ± 0.06 | 0.021 ± 0.017 | 170.30 ± 72.85 |
| rain_0.25 | 13/20 (65.0%; 95% CI 43.3%–81.9%) | 7/20 (35.0%) | 0 | 0 | 7.98 ± 0.37 | 0.25 ± 0.08 | 0.018 ± 0.015 | 162.92 ± 70.31 |
| rain_0.75 | 10/20 (50.0%; 95% CI 29.9%–70.1%) | 8/20 (40.0%) | 2 | 0 | 8.04 ± 0.37 | 0.29 ± 0.06 | 0.021 ± 0.017 | 169.90 ± 71.46 |

## YOLO detection-instance ablation

Counts are raw predicted box instances accumulated over episode frames, not unique tracked vehicles. No annotated ground truth is available, so these values cannot be interpreted as detection accuracy, precision, or recall. Frame-normalized values account for different episode lengths.

| Weather | Vehicle boxes / frame | Vehicle box instances | Car | Truck | Bus | Motorcycle | Person | Frames with any box | Mean stereo valid fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clear_0.00 | 1.619 | 10644 | 4689 | 5919 | 12 | 24 | 106 | 32.9% | 0.0101 |
| fog_0.25 | 1.558 | 9926 | 3965 | 5949 | 12 | 0 | 59 | 34.5% | 0.0101 |
| fog_0.75 | 1.215 | 7833 | 3048 | 4777 | 8 | 0 | 20 | 28.3% | 0.0100 |
| rain_0.25 | 1.904 | 12831 | 5053 | 7653 | 124 | 1 | 228 | 47.7% | 0.0079 |
| rain_0.75 | 0.907 | 5701 | 1810 | 3444 | 447 | 0 | 273 | 32.3% | 0.0045 |

## YOLO box instances by camera

| Weather | Front left | Left | Right | Rear |
|---|---:|---:|---:|---:|
| clear_0.00 | 3288 | 2477 | 2477 | 2508 |
| fog_0.25 | 3046 | 2309 | 2309 | 2321 |
| fog_0.75 | 2291 | 1854 | 1854 | 1854 |
| rain_0.25 | 3728 | 3084 | 3082 | 3165 |
| rain_0.75 | 1672 | 1442 | 1394 | 1466 |

## Interpretation limits

- This is a simulator pilot and evaluates a checkpoint trained for only 2,048 PPO timesteps; it does not establish convergence or deployment safety.
- The seeds are held out from the training scenario-index range, but map/scenario diversity is limited to MetaDrive's generated scenario pool.
- Synthetic weather changes camera pixels and supplies a condition-derived visibility horizon to BEV construction; it does not simulate rain/fog effects on tire-road dynamics.
- Energy is an idealized flat-road tractive-work proxy, not measured battery energy; regeneration, grade, and drivetrain efficiency are omitted.
- YOLO counts are predictions and can include repeated observations of the same vehicle across frames.

Episode-level source: `bev_policy/checkpoints/ppo_bev_adaptive_safety_v1.zip` and per-condition JSON/CSV files in `E:\Capstone\bevcapstone\bev_policy\outputs\ppo_weather_generalization_20_stereo05`.
