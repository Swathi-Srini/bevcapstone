# Geometric corrections in the camera grid builder (complete record)

File: `bev_state/camera_grid_builder.py` (ours; frozen by hash; the only system module we modify). Written 14 Nov 2026. Companion: `docs/FUNCTION_MAP.md` (call chain and function-level map), `report/latest_14th_Nov_26.pdf` (narrative), `report/Final_Methodology_and_Approach.pdf`.

## 1. What the builder does, in one paragraph

The unmodified perception chain reports each detected vehicle as a *point* in the ego frame (`ObjectFootprint`: label, `x_right_m`, `y_forward_m`, width, length, confidence, camera, depth method). The ground-truth grid, in contrast, draws one true-size rectangle per vehicle at its centre. The builder converts the first into the second **using only the detections and the camera that produced them**; it never reads simulator vehicle positions. It then hands the corrected objects to the **original, unmodified** rasteriser (`BEVStateAssembler.build_bev_grid`) and adds the HD-map layers with the same function that adds them to the ground-truth grid (`overlay_github_bev`).

## 2. Where each correction lives (function, line, order of application)

Per detection, in this order (all in `bev_state/camera_grid_builder.py`):

| Order | Function (line) | Correction | Applies to |
|---|---|---|---|
| 1 | `CameraGridBuilder.correct` (L58), branch `left_camera`/`right_camera` | **Point reflection** `(x, y) -> (-x, -y)` | side-camera objects |
| 2 | same, constants `SIDE_AFFINE` (L37) / `FRONT_AFFINE` (L36) | **Affine centre correction** `[x, y, 1] -> centre` | side (after reflection) and front stereo objects |
| 2b | same, branch `rear_camera` | half-footprint shift along the line of sight (geometric, **unvalidated**) | rear objects |
| 3 | same, final `replace(...)`, constants `VEHICLE_WIDTH_M, VEHICLE_LENGTH_M` (L38) | **True-size footprint** 2.0 x 4.9 m | every object |
| 4 | `CameraGridBuilder.merge_duplicates` (L71), called by `corrected_objects` (L79) | **One object per vehicle**: non-maximum suppression in the BEV plane | all objects together |
| 5 | `CameraGridBuilder.build` (L83) | rasterise with `asm.build_bev_grid`, then `overlay_github_bev` | the grid |

`corrected_objects` is also called by `PilotEnv.grid_and_metrics` (`bev_rl/adaptive/pilot_camera_builder.py`) so the pilot could score corrected objects separately from the raw ones. `build` is called by `BuilderEnv.student_bev` in `bev_rl/adaptive/camera_dagger.py` (training and validation) and by `PilotEnv.grid_and_metrics` (pilot and evaluation).

## 3. Each correction: problem, evidence, formula, status

### 3.1 Side cameras are point-reflected
* **Problem.** A vehicle on the ego's right is rendered by `left_camera` (and vice versa), but the unmodified position conversion (`ego_position_from_camera` / `estimate_detection_position` in `YOLO+weather module/manual_drive_stereo_yolo_weather.py`) assumes the opposite orientation. The reported `(x, y)` is the true position reflected through the ego origin (the cameras sit on the ego origin).
* **Evidence.** In all 53 side detections of the calibration scenes the report was closer to the *reflected* truth than to the truth; median error 11.4 m before, 2.7 m after the reflection alone.
* **Correction.** `x, y = -x, -y` before anything else.
* **Why in our module and not at the source.** Only the builder may be changed. The cause is in the unmodified conversion; a source fix would be cleaner and is a decision for the owner of that module.
* **Status.** Validated on the calibration scenes only (three scenes).

### 3.2 Surface point is not the vehicle centre (affine correction)
* **Problem.** Front: stereo depth is the median of valid pixels in the central part of the box, i.e. the *nearest visible surface*, so the reported point lies short of the centre; truncation and median-depth biases add to it. Side: the ground-plane projection uses the bottom edge of the box, again the near surface.
* **Correction.** A least-squares affine map `[x_reported, y_reported, 1] -> [x_true_centre, y_true_centre]`, one per camera group:
  * `FRONT_AFFINE = [[1.15, 0.02, -0.07], [0.11, 1.03, 0.87]]` (front-left camera, stereo objects)
  * `SIDE_AFFINE = [[1.20, -0.04, -1.50], [0.02, 1.67, 0.27]]` (left and right cameras, applied after the reflection)
  * rows give `x_true` then `y_true`.
* **Rear camera.** One sample only, so no fit: the object is shifted outward by half the footprint along the line of sight (`s = min((w/2)/|x/rho|, (l/2)/|y/rho|)`). **Unvalidated.**
* **Reported effect** (leave-one-scene-out, calibration data): median error front 1.58 -> 0.91 m; side 11.36 -> 2.71 (reflection) -> 1.30 m (reflection + affine).

### 3.3 True-size footprint
* **Problem.** YOLO class labels are unreliable for size (a 4.6 m car is reported as `car` or `bus`), and the assembler's class priors / bbox clamp (`estimate_footprint`) therefore give wrong sizes.
* **Correction.** Every object gets the mean true traffic-vehicle footprint **2.0 x 4.9 m** (`VEHICLE_WIDTH_M, VEHICLE_LENGTH_M`), because the ground-truth grid draws the true rectangle. The constant is the mean simulator traffic-vehicle size, read from the simulator once, offline.
* **How the rasteriser draws it.** `_draw_oriented_footprint` (unmodified, `assembler.py` L309) draws an **axis-aligned** rectangle (no heading) centred on the cell of `(x, y)`: width `round(2.0/0.3125) = 6` cells, length `round(4.9/0.3125) = 16` cells.

### 3.4 One object per vehicle (duplicate merge)
* **Problem.** The same vehicle can be reported by several cameras and by overlapping boxes; the original builder then drew several rectangles (6 duplicates in the clear-weather pilot).
* **Correction.** After correction, objects closer than **3.5 m forward and 1.6 m laterally** (`CameraBuilderConfig.dedup_forward_m`, `dedup_lateral_m`) are merged; stereo-measured objects and higher-confidence objects win (`sorted(... key=(depth_method == "stereo_sgm", confidence), reverse=True)`). The thresholds were chosen by hand.
* **Effect.** Duplicates 6 -> 0 in the pilot; false objects 32 -> 6 in clear weather.

### 3.5 What is deliberately not changed
Rasterisation (`build_bev_grid`), the free-space mask (`_mark_visible_free_space`: the assembler's camera-cone geometry, computed once and copied by `CautionEnv._cache_free_space`; it does not depend on the picture or the weather), the HD-map layers, depth, detection, the weather filter, the camera rig.

## 4. The calibration experiment (where the constants came from)

* **Script:** `bev_rl/adaptive/calibrate_camera_geometry.py`. Per scene seed, K = 2 traffic vehicles are placed **one at a time** at every point of a grid (lateral x in {-8, -5.5, -3.5, 0, 3.5, 5.5, 8} m, forward y in {-2, 1, 4, 7, 10, 13, 16} m, lane-aligned, stationary); all other vehicles are moved 300 m away; the ego does not move. Clear weather (geometry is calibrated without weather).
* **What runs:** the full unmodified chain on the five rendered pictures: `asm.assemble(frames=...)` = YOLO on four views + front stereo + `objects_from_detections`.
* **Stored per placement:** the true centre `(true_x, true_y)` in the ego frame, the true `length` and `width`, and every object the pipeline returned (camera, depth method, label, confidence, reported x, y). Files: `bev_rl/adaptive/results/camera_geometry/placements_seed{2000,2001,2002,3100,3101,3102}.jsonl` (98 placements each; 294 in the three fit scenes).
* **Ground truth used:** the placed vehicle's true centre and size, read from the simulator, **offline only**, to fit the constants. The builder never sees it at run time.
* **Seeds:** 2000-2002 are the fit scenes. 3100-3102 were collected as a check; whether they entered the original fit cannot be proven from the repository (see section 5). Seed 3100 is contaminated (a parked truck displaced the placed vehicle by 30-45 m).
* **Recorded result (from the original run):** 85 usable point pairs (front 52, side 33); leave-one-scene-out errors as in section 3.2.

## 5. Reproducibility audit (14 Nov 2026) -- an honest gap

The original fitting script was **not saved** in the repository, so the constants could not be reproduced from it. `bev_rl/adaptive/fit_camera_corrections.py` (new, numpy only, no simulator) re-derives the corrections from the stored placement files with a documented pairing rule: per placement and camera group take the reported object nearest to the true centre (side: after the reflection), keep the pair if it is within 6 m and the true centre is inside the BEV window.

| | Recorded (original run) | Re-derived (this audit, seeds 2000-2002) |
|---|---|---|
| Pairs | 85 (front 52, side 33) | 72 (front 49, side 23) |
| Front coefficients, max difference to the builder's constants | -- | 0.084 (0.019 with a 3 m pairing radius) |
| Front error before -> after (leave-one-scene-out) | 1.58 -> 0.91 m | 1.61 -> 0.95 m |
| Side coefficients, max difference to the builder's constants | -- | **0.227** (cross terms differ: -0.04 vs +0.14) |
| Side error before reflection+affine -> after | 2.71 (after reflection) -> 1.30 m | 2.66 -> 1.04 m |

Sixteen pairing variants (window on/off, radius 3-12 m, nearest object or all, reflect before or after measuring the distance) and adding the 3100-3102 files were tried; **none reproduces 52 / 33**. Conclusions:

1. The **front** correction is reproducible to within about 0.1 per coefficient and gives the same error reduction.
2. The **side** correction is only approximately reproducible; the recorded constants cannot be re-derived exactly. The side fit used fewer well-posed pairs (23-33) than the front.
3. The constants in the builder are **unchanged**, were used in the pilot and are frozen by hash; nothing was refitted. In any write-up, describe the fit as "least squares on controlled-placement pairs from three scenes (about 72-85 pairs)" and quote the re-derived numbers, with the discrepancy stated.
4. A better calibration (more scenes, more distances, a saved script) would remove this weakness; it is not on the current schedule.

## 6. Measured effect of the corrections (pilot v2, 10 episodes per weather; builder v2 versus the original builder)

| | Clear | Fog 1.0 | Rain 0.6 | Rain 1.0 |
|---|---|---|---|---|
| Recall within 3 m (original -> v2) | 0.351 -> 0.474 | 0.002 -> 0.019 | 0.080 -> 0.082 | 0 -> 0 |
| Median position error (m) | 1.41 -> 0.93 | 0.84 -> 0.49 | 2.08 -> 1.09 | -- |
| Duplicates | 6 -> 0 | 0 -> 0 | 0 -> 0 | 0 -> 0 |
| False objects | 32 -> 6 | 7 -> 1 | -- -> 2 | 0 -> 0 |
| Vehicle-cell IoU with ground truth | 0.136 -> 0.278 | 0.003 -> 0.013 | 0.069 -> 0.064 | 0 |
| Exact BEV equality on vehicle frames | 0/82 | 0/180 | 0/216 | 0/177 |

The corrections cannot add a vehicle the detector did not report (YOLO misses large side-on vehicles at 4-7 m and edge-truncated ones; recall collapses in rain; stereo drops boxes without valid depth), so exact equality on vehicle frames is not achievable.

## 7. Limits (state in any write-up)
* Calibrated in **clear weather**, on **three scenes**, with **stationary, lane-aligned, single vehicles**; applied unchanged in fog and rain, to moving traffic and to other orientations.
* The rear-camera correction is unvalidated. The reflection and the affine terms compensate for an integration issue in the unmodified modules rather than fixing it at the source.
* The footprint size and merge thresholds are hand-chosen constants.
* The side fit is only approximately reproducible (section 5).

## 8. How to check things yourself
* Corrections at work: `python -m bev_rl.adaptive.pilot_camera_builder ...` (writes both raw and corrected metrics per step).
* Re-derive the constants: `python -m bev_rl.adaptive.fit_camera_corrections`.
* Hash of the builder in the freeze manifest: `bev_rl/adaptive/results/camera_dagger_final/FROZEN_MANIFEST.json`.
* Related integrity check: `YOLO+weather module/yolo/yolo_utils.py` contains a `FallbackDetector` that would project simulator vehicles into the image; it is **never instantiated anywhere**, `ensure_yolo_model` raises an error instead of falling back, and the worker logs show `Loading pretrained YOLO model: weights/yolov8n.pt`. No privileged detections enter the camera pipeline.
