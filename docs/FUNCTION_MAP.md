# Function-level map of the current pipeline (14 Nov 2026)

What each active file is for, which functions are used where, and why. Companion: `docs/REPO_MAP.md` (all files by category), `docs/GEOMETRIC_CORRECTIONS.md` (the builder's corrections). Line numbers are those of 14 Nov 2026. "Called by" was checked from the code (AST call graph plus grep).

## 1. One simulation step during camera DAgger collection

```text
collect()  [bev_rl/adaptive/camera_dagger.py:148]
 |-- env.student_bev()   = BuilderEnv.student_bev  [camera_dagger.py:47]   <- builds the student's BEV from the cameras
 |     |-- frame_from_sensor(md, name) x5        [YOLO+weather module/manual_drive_stereo_yolo_weather.py:74]  render 5 RGB frames
 |     |-- apply_weather(frame, kind, level) x5  [YOLO+weather module/weather/weather_utils.py:49]  (add_fog / add_rain)
 |     |-- apply_synchronized_stereo_weather()   [manual_drive_stereo_yolo_weather.py:89]  same random draw on the stereo pair
 |     |-- A.front_stereo_depth(L, R, asm.matcher) [manual_drive_stereo_yolo_weather.py:105]  StereoSGBM, Z = f*B/d
 |     |-- A.run_yolo(yolo, frame) x4            [YOLO+weather module/yolo/yolo_utils.py:537]  YOLOv8n on front-left, left, right, rear
 |     |-- asm.objects_from_detections(dets, depth) [bev_state/assembler.py:175]  boxes -> ego-frame points
 |     |     |-- estimate_detection_position()   [manual_drive_stereo_yolo_weather.py:163]  stereo (front) / ground plane (side, rear)
 |     |     |     |-- object_stereo_depth()     [:124]   median valid depth in the box centre
 |     |     |     `-- ego_position_from_camera()[:151]   pixel + depth -> (x right, y forward)
 |     |     `-- estimate_footprint()            [assembler.py:204]  class-prior size (overridden by our builder)
 |     `-- builder.build(objects, md)            [bev_state/camera_grid_builder.py:83]   <== OUR MODULE
 |           |-- corrected_objects()             [:79] -> correct() [:58] per object, merge_duplicates() [:71]
 |           |-- asm.build_bev_grid(corrected)   [assembler.py:225]  original rasteriser
 |           |     |-- _mark_visible_free_space  [assembler.py:270]  (replaced by a cached copy; identical result)
 |           |     `-- _draw_oriented_footprint -> ego_to_grid [assembler.py:309 / 260]  axis-aligned rectangle
 |           `-- overlay_github_bev(grid, md)    [bev_rl/adaptive/spec_bev.py:66] -> map_layers [:38] -> _lane_chain [:26], _to_cells [:19]
 |-- raw = o[:6] * SCALE      scalars come from AdaptiveEnv._obs [bev_rl/adaptive/env.py:65] -> asm.extract_scalar_state [assembler.py:340]
 |-- ea = act_teacher(env)    = teacher_driver().act [camera_dagger.py:96]
 |     |-- curvature_cap_mps()  [bev_rl/adaptive/teacher_eval.py:27]   curve guard (privileged HD-map geometry)
 |     `-- SafeTeacher.act()    [bev_rl/adaptive/safe_teacher.py:43] -> ttc [:25] -> _extent [:19]   (IDM from build_teacher [teacher_eval.py:42])
 |-- act = ea (teacher drives) or student_act() [camera_dagger.py:85] -> prep() [bev_rl/adaptive/caution.py:88] -> BEVScalarPolicy [bev_policy/model.py]
 `-- env.step(act)            [bev_rl/adaptive/env.py:88]  MetaDrive step, energy via step_work_j [bev_rl/energy.py:71], outcome
```

After the episode: `np.savez_compressed(...)` stores grids (int8 x10), scalars, clean teacher actions, caution values. `train()` [camera_dagger.py:185] retrains on all rounds with the `KEEP` mask from `caution.py:26`; `validate()` [:207] and `select()` [:231] choose the checkpoint.

## 2. Files and their purpose in the current work

| File | Purpose | Owner |
|---|---|---|
| `bev_state/assembler.py` | the contract: `ObjectFootprint`, `objects_from_detections`, `build_bev_grid` (rasteriser), `extract_scalar_state`; imports the perception modules | teammates, unmodified |
| `bev_state/camera_grid_builder.py` | **our** geometric corrections between detections and rasteriser | ours, frozen |
| `YOLO+weather module/manual_drive_stereo_yolo_weather.py` | camera rig, frame capture, synchronised weather, stereo depth, detection-to-position | teammates |
| `.../weather/weather_utils.py`, `.../yolo/yolo_utils.py` | the weather filter; YOLO loading and inference | teammates |
| `bev_rl/adaptive/camera_dagger.py` | DAgger trainer (collect, train, validate, select, freeze); `BuilderEnv` | ours |
| `bev_rl/adaptive/e2e_pipeline.py` | `E2EEnv`: base class (builds the 5-camera MetaDrive, loads YOLO, mounts the rig per reset) | ours |
| `bev_rl/adaptive/env.py` | `AdaptiveEnv`: the Gym-style environment (step, energy, outcomes, scalars, `truth_objects`) | ours |
| `bev_rl/adaptive/caution.py` | `CautionEnv`, `keep_prob`, `c_auto`, `expert_params`, `prep`, `KEEP` mask | ours |
| `bev_rl/adaptive/spec_bev.py` | HD-map layers (`overlay_github_bev`), lane chain | ours |
| `bev_rl/adaptive/safe_teacher.py`, `teacher_eval.py` | collision-aware teacher; teacher construction, curve guard, validation | ours |
| `bev_rl/adaptive/pilot_camera_builder.py` | `PilotEnv`: full-pipeline run with per-step scoring against ground truth | ours |
| `bev_rl/adaptive/evaluate_camera_pipeline.py`, `summarise_camera_eval.py` | final camera evaluation (arms, tasks, videos); tables and Wilson intervals | ours (never run yet) |
| `bev_rl/adaptive/calibrate_camera_geometry.py`, `fit_camera_corrections.py` | controlled placements; offline re-derivation of the corrections | ours |
| `bev_policy/model.py`, `bev_rl/energy.py` | the network; the energy model | teammates / ours |

## 3. Function tables (what, why, who calls it)

### `bev_state/camera_grid_builder.py`
| Function | Line | What | Why | Called by |
|---|---|---|---|---|
| `CameraGridBuilder.correct` | 58 | reflect (side), affine (front/side), half-footprint shift (rear), set true-size footprint | the reported point is a reflected / nearest-surface point of unknown size | `corrected_objects` |
| `.merge_duplicates` | 71 | non-maximum suppression in the BEV plane | one rectangle per vehicle | `corrected_objects` |
| `.corrected_objects` | 79 | `merge_duplicates([correct(o) ...])` | single entry point; also used to score corrected objects separately | `build`; `PilotEnv.grid_and_metrics` |
| `.build` | 83 | corrected objects -> `asm.build_bev_grid` -> `overlay_github_bev` | the grid the policy sees | `BuilderEnv.student_bev` (training, validation); `PilotEnv.grid_and_metrics` (pilot, evaluation) |

### `bev_state/assembler.py` (functions our code uses)
| Function | Line | What | Why we use it | Called by |
|---|---|---|---|---|
| `objects_from_detections` | 175 | YOLO boxes + depth -> `ObjectFootprint` list (drops boxes without valid depth) | the original detection-to-position stage | `BuilderEnv.student_bev`, `PilotEnv` |
| `estimate_footprint` | 204 | class-prior width/length with a bbox clamp | inside `objects_from_detections` (our builder later overrides the size) | `objects_from_detections` |
| `build_bev_grid` | 225 | rasterise objects into the 64x64 grid | the original rasteriser; used for the camera grid, the ground-truth grid and the pilot's comparison grids | `CameraGridBuilder.build`, `AdaptiveEnv.bev`, `CautionEnv.student_bev`, `PilotEnv` |
| `_mark_visible_free_space` | 270 | free cells inside the four camera cones within 30 m | the original free-space rule; cached per instance by `CautionEnv._cache_free_space` (bit-identical) | `build_bev_grid` |
| `_draw_oriented_footprint` / `ego_to_grid` | 309 / 260 | draw an axis-aligned rectangle at the cell of (x, y) | the original footprint drawing | `build_bev_grid` |
| `extract_scalar_state` | 340 | the 6-D ego state (speed, progress, lateral, heading, curvature, goal) | scalars for the policy | `AdaptiveEnv._obs` |
| `assemble` | 118 | the whole chain in one call | **not used** in training/evaluation; used by `calibrate_camera_geometry.py` and the superseded e2e files | -- |

### `YOLO+weather module/manual_drive_stereo_yolo_weather.py`
| Function | Line | What | Called by |
|---|---|---|---|
| `build_stereo_matcher` | 63 | StereoSGBM with the specified parameters | `BEVStateAssembler.__init__` |
| `frame_from_sensor` | 74 | read a camera's RGB array (BGR, uint8) | `BuilderEnv.student_bev`, `PilotEnv`, `calibrate_camera_geometry.main` |
| `mount_camera_rig` | 81 | attach the five cameras, set the 60 deg lens | `E2EEnv.reset`, `assembler.mount_if_needed` |
| `apply_synchronized_stereo_weather` | 89 | identical random draw on the stereo pair (saves/restores the random state) | `BuilderEnv.student_bev`, `PilotEnv` |
| `front_stereo_depth` | 105 | disparity -> depth, valid range 1-30 m, speckle filter | `BuilderEnv.student_bev`, `PilotEnv` |
| `object_stereo_depth` | 124 | median of valid depth pixels in the central part of the box | `estimate_detection_position` |
| `ego_position_from_camera` | 151 | pixel column + depth -> ego (x, y) using the camera yaw | `estimate_detection_position` |
| `estimate_detection_position` | 163 | stereo for the front camera, ground-plane (Z = H/tan(phi)) for side and rear | `objects_from_detections` |

### `bev_rl/adaptive/camera_dagger.py`
| Function | Line | What | Why |
|---|---|---|---|
| `make_env` / `BuilderEnv.student_bev` | 41 / 47 | the camera-pipeline observation (section 1); `light=True` (unused now) would skip perception | student observation for every step; the old skip-step shortcut was removed |
| `load_net` / `student_act` | 79 / 85 | load a checkpoint with its normalisation; one student forward pass | initialisation from the original checkpoint; acting |
| `teacher_driver` | 91 | build the privileged teacher (`teacher_eval.build_teacher`) and apply the curve guard each step | clean action labels |
| `round_tasks` | 103 | deterministic (density, seed, weather, c) list per round | reproducible rounds; seeds from 2000-2999 |
| `claim` | 116 | atomic claim file per episode | lets several workers share a round and resume |
| `freeze` / `verify_frozen` / `_sha` | 131 / 138 / 126 | hash manifest of the frozen files; abort if any changed | integrity of the experiment |
| `collect` / `train` / `validate` / `select` | 148 / 185 / 207 / 231 | the DAgger stages | the approved recipe |

### `bev_rl/adaptive/env.py`, `caution.py`, `e2e_pipeline.py`
| Function | What | Why / who |
|---|---|---|
| `AdaptiveEnv.reset/step` (env.py:70/88) | MetaDrive reset/step, outcome, energy via `step_work_j` | the environment every path uses |
| `AdaptiveEnv.truth_objects` (116) | simulator vehicles in the window as `ObjectFootprint` | **ground truth**: used by the teacher, by scoring, and by the *old* simulator-grid evaluation; never by the camera student |
| `AdaptiveEnv.bev` (134) | the ground-truth grid via `build_bev_grid` | the reference grid for comparison |
| `AdaptiveEnv._obs` (65) | normalised scalar vector via `extract_scalar_state` | the student's scalars |
| `CautionEnv._cache_free_space` (caution.py:58) | compute the free-space mask once, copy afterwards | speed-up, bit-identical |
| `CautionEnv.student_bev` (74) | **old** student grid: truth objects dropped by `keep_prob`, then rasterised | the 200-episode simulator-grid path; **overridden** by `BuilderEnv.student_bev` |
| `keep_prob`, `c_auto` (29, 37) | old retention table -> keep probability, caution | used by the original caution model and the old study; retired for the new method |
| `expert_params` (43) | caution -> IDM target speed and gap | Teacher B's mapping |
| `prep` (88) | normalise scalars with the KEEP mask, build tensors | every student forward pass |
| `E2EEnv.__init__/reset` (e2e_pipeline.py) | build the 5-camera MetaDrive (the first MetaDrive is closed), load YOLO, mount the rig after every reset | base class of `BuilderEnv` and `PilotEnv` |
| `E2EEnv.student_bev/raw_bev` | call `asm.assemble()` | **superseded**, used only by `e2e_dagger*.py` and `evaluate_e2e.py` |

### Teachers
| Function | File:line | What |
|---|---|---|
| `build_teacher`, `_build` | `teacher_eval.py:42, 49` | IDM + SafeTeacher + caution parameters by name (`stock_idm`, `safe`, `caut_idm`, `cs0`, `cs1`, `_g<a_lat>`) |
| `curvature_cap_mps` | `teacher_eval.py:27` | curve guard speed cap from the HD-map geometry (uses `spec_bev._lane_chain`) |
| `SafeTeacher.act`, `.ttc`, `._extent` | `safe_teacher.py:43, 25, 19` | constant-velocity TTC and graded braking (1.3 s / 2.4 s) |

### Evaluation and analysis
| Function | File:line | What |
|---|---|---|
| `PilotEnv.grid_and_metrics` | `pilot_camera_builder.py:65` | one step of the pipeline plus scoring against ground truth (raw and corrected objects, grid equality, IoU); also records `last_front` for videos |
| `match`, `in_view` | `pilot_camera_builder.py:32, 48` | greedy 3 m matching; field-of-view test |
| `main`, `parse_tasks`, `load_arm`, `video_frame` | `evaluate_camera_pipeline.py:60, 32, 40, 49` | the final camera evaluation loop (arms, task list, checkpoints, video frames) |
| `main`, `wilson` | `summarise_camera_eval.py:23, 18` | tables with Wilson intervals, completion time, speed, energy, perception metrics |
| `load_model`, `row` | `evaluate_n200_all_weather.py:38, 33` | load the original checkpoints; per-episode result row (reused by the camera evaluation) |
| `main` | `calibrate_camera_geometry.py:23` | controlled placements for the corrections |
| `pairs`, `fit`, `loso`, `main` | `fit_camera_corrections.py` | offline re-derivation of the corrections |

## 4. Retired / not on the path
`visibility_sensor.py` (`VisibilitySensor.update`, `features`, `fit`): the retired image-statistics sensor; imported lazily by `camera_dagger.make_env` only when `sensor=True` (the `caut` model entry); OV-Split does not use it; its file hash is in the freeze manifest. `FallbackDetector` in `yolo_utils.py`: never instantiated.
