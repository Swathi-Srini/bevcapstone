# Repository code map (14 November 2026)

Which code is used by the current work, which is finished, which is historical, which belongs to the reinforcement-learning part, and which is superseded. **Nothing here has been moved or deleted**; this is a map. The descriptions are the first line of each file's own docstring or header. For the narrative see `report/latest_14th_Nov_26.pdf`; the short version is the root `README.md`.

**Project title:** *Reinforcement learning under fog and rain conditions.* **Current work (this map's focus):** proper *base models* trained on our own perception pipeline (OV-Split Camera and Live Caution Camera), before any RL fine-tuning. **Reinforcement learning is a separate part, handled by a teammate**; its code and what was tried are listed in section 4 and in the README's RL section.

## 0. How the current training uses `bev_state/assembler.py`

Yes, it is used, but not through `assemble()`. The camera DAgger trainer (`bev_rl/adaptive/camera_dagger.py`, class `BuilderEnv`) replicates the assembler's steps so that our builder can sit between the detections and the rasteriser:

| Step | Code | From |
|---|---|---|
| Weather filter on all 5 pictures; synchronised noise on the stereo pair | `apply_weather`, `apply_synchronized_stereo_weather`, `frame_from_sensor`, `mount_camera_rig` | `YOLO+weather module/` (teammates', unchanged; imported by `assembler.py`) |
| Front stereo depth | `A.front_stereo_depth(...)` with `asm.matcher` (StereoSGBM) | `assembler.py` re-exports it from `manual_drive_stereo_yolo_weather.py` |
| YOLOv8n on 4 views | `A.run_yolo(...)`, `A.YOLO_CAMERAS` | `assembler.py` / `yolo/yolo_utils.py` |
| Detection to ego position and objects | `asm.objects_from_detections(dets, depth)` (stereo for front, ground-plane for side and rear) | **`assembler.py`, unmodified** |
| Corrections, true-size footprint, duplicate merge | `CameraGridBuilder.corrected_objects` | **`bev_state/camera_grid_builder.py` (ours)** |
| Rasterisation into the 64x64 grid | `asm.build_bev_grid(corrected_objects)` | **`assembler.py`, unmodified.** The free-space mask is the assembler's own camera-cone routine, computed once and copied by `CautionEnv._cache_free_space` (bit-identical); it depends on the camera geometry, not on the picture |
| HD-map layers (centreline 0.5, boundary 0.8) | `overlay_github_bev` | `bev_rl/adaptive/spec_bev.py` |
| Ego scalars | `asm.extract_scalar_state(md, info)` (called in `bev_rl/adaptive/env.py`) | `assembler.py` |

Not used in the current training: `assembler.assemble()` (used only by the superseded `e2e_pipeline.student_bev`, `e2e_dagger*.py` and `evaluate_e2e.py`) and the privileged `visibility_range_m` masking option.

## 1. Active: used by the running or queued work (do not touch while OV-Split trains)

These are imported by the running workers every round, or hashed in `results/camera_dagger_final/FROZEN_MANIFEST.json` (builder, assembler, `bev_state/__init__.py`, `visibility_sensor.py`, both original checkpoints, `yolov8n.pt`, the old retention table). Moving or editing them stops training, because workers abort if a hash changes.

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_state/assembler.py` | Assemble policy-ready BEV and ego-state observations. |
| `bev_state/__init__.py` | BEV state assembly contract for the MetaDrive project. |
| `bev_state/camera_grid_builder.py` | Camera-side BEV grid builder, version 2 (geometry and representation corrections only). |
| `YOLO+weather module/manual_drive_stereo_yolo_weather.py` | Live four-direction camera rig: front stereo depth + YOLO + weather. |
| `YOLO+weather module/weather/weather_utils.py` | Weather augmentation helpers for fog and rain effects. |
| `YOLO+weather module/yolo/yolo_utils.py` | YOLO helper functions for detection and annotation. |
| `bev_policy/model.py` | Policy architecture: BEV CNN + six-dimensional state MLP. |
| `bev_policy/runtime.py` | Shared MetaDrive configuration for the BEV BC collector and evaluator. |
| `bev_rl/energy.py` | Estimated mechanical traction energy from simulated longitudinal speed. |
| `bev_rl/telemetry.py` | Per-step vehicle telemetry read from MetaDrive (training/evaluation signals only). |
| `bev_rl/reward.py` | Step reward for clear-weather, zero-traffic route following. |
| `bev_rl/adaptive/camera_dagger.py` | Camera-pipeline DAgger for the two policies, as SEPARATE loops with separate datasets and checkpoints. |
| `bev_rl/adaptive/e2e_pipeline.py` | END-TO-END environment: the bird's-eye-view grid is built from the CAMERAS, not from the simulator's vehicle list. |
| `bev_rl/adaptive/env.py` | Fast camera-free MetaDrive environment for weather-adaptive, energy-aware driving (privileged teacher). |
| `bev_rl/adaptive/caution.py` | Reliability-Coupled Caution Conditioning (policy-side contribution). |
| `bev_rl/adaptive/spec_bev.py` | Spec-compliant BEV overlay (docs/AV_Carla_BEV.pdf, Sec. 5): HD-map route centreline and road boundaries. |
| `bev_rl/adaptive/safe_teacher.py` | Collision-aware teacher: MetaDrive IDM (lane keeping + same-lane car following) + a privileged predictive safety layer. |
| `bev_rl/adaptive/teacher_eval.py` | Evaluate TEACHERS on their own (simulator only; teachers read the exact simulator state, no camera is involved). |
| `bev_rl/adaptive/evaluate_camera_pipeline.py` | FULL CAMERA-PIPELINE EVALUATION of policies (matched seeds, identical conditions for every model), with optional front-camera videos. |
| `bev_rl/adaptive/pilot_camera_builder.py` | PILOT: drive the ORIGINAL proposed model (ov_split, unchanged checkpoint) from the grid built by the full camera pipeline + the new CameraGridBuilder, |
| `bev_rl/adaptive/summarise_camera_eval.py` | Aggregate camera-pipeline evaluation episodes (read-only): outcomes with Wilson 95% intervals, completion time, speed, estimated energy, and perceptio |
| `bev_rl/adaptive/evaluate_n200_all_weather.py` | Final n=200 ALL-WEATHER evaluation, ONE harness for all six presented models (records energy for every model). |
| `bev_rl/adaptive/visibility_sensor.py` | LIVE caution sensor: estimates the caution level c in [0,1] from the camera picture alone, every step. No weather label, no simulator state. |
| `bev_rl/adaptive/queue_camera_training.sh` | One camera-pipeline DAgger loop (rounds 0-5, 40 episodes per round, density 0.05) for ONE model:  queue_camera_training.sh ov/caut |
| `bev_rl/adaptive/queue_pack.sh` | DELIVERABLE PACK (queued): ORIGINAL ("200-episode", ground-truth-trained) ov_split and caution checkpoints on the FULL camera pipeline. |
| `bev_rl/adaptive/queue_students_eval.sh` | QUEUED: camera-trained OV-Split student (the model being trained now) on the FULL camera pipeline: fog 1.0 seeds 5100-5109 (10 episodes) + front-camer |
| `bev_rl/adaptive/queue_camera_eval.sh` | Queue entry: full camera-pipeline evaluation of the ORIGINAL OV-Split and caution checkpoints (80 episodes: 10 seeds x 4 weathers x 2 models). |


Also used at run time through `sys.path` injection by `bev_state/assembler.py`: `YOLO+weather module/` (`manual_drive_stereo_yolo_weather.py`, `weather/weather_utils.py`, `yolo/yolo_utils.py`, `stereo_depth/`), the teammates' modules, unchanged. `camera_dagger.py` still contains a `caut` entry that uses the **retired** image sensor; it is not used (Live Caution will get a new trainer after OV-Split finishes).

## 2. Built during the camera-pipeline effort, finished (kept as record and for reproduction)

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_rl/adaptive/calibrate_camera_geometry.py` | CONTROLLED PLACEMENT experiment: what does the unmodified camera pipeline report for a vehicle at a KNOWN position around the ego? |
| `bev_rl/adaptive/compare_grids.py` | Are the grids of the 200-episode tests the SAME as the grids the full camera pipeline produces? |
| `bev_rl/adaptive/queue_pilot_camera_builder.sh` | PILOT of the camera grid builder: original proposed checkpoint (ov_split, unchanged), 10 episodes (seeds 5000-5009) per weather (clear, fog 1.0, rain  |
| `bev_rl/adaptive/queue_pilot_camera_builder_v2.sh` | PILOT v2 (builder with the side-camera reflection fix, fitted geometry, true-size footprint): original ov_split checkpoint, seeds 5000-5009 x {clear,  |
| `bev_rl/adaptive/queue_teacher_validation.sh` | Teacher validation (simulator only, no cameras). Seeds 3100-3199, density 0.05, horizon 1600 (the 80 s result is read off the same episodes: success w |
| `bev_rl/adaptive/queue_teacher_guard.sh` | Curvature-guard variants of the SafeTeacher on the validation seeds (design only; seeds 3100-3199, clear = weather-invariant for a privileged teacher  |
| `bev_rl/adaptive/queue_teacher_guard_caution.sh` | Caution-SafeTeacher (fixed thresholds, cs0) WITH the curvature guard at 4.0 m/s^2, validation seeds 3100-3199, density 0.05, horizon 1600. |
| `bev_rl/adaptive/queue_teacher_density.sh` | Teacher results per traffic density (simulator only, no camera). ONE sequential process. Output: results/teacher_density (new folder; nothing existing |
| `bev_rl/adaptive/summarise_teacher_density.py` | Per-density teacher table from results/teacher_density/*.json (50 validation seeds 3100-3149, horizon 160 s). |
| `bev_rl/adaptive/evaluate_visibility_sensor.py` | SENSOR VALIDATION (not tuning): apply the FROZEN weights (results/visibility_sensor/weights.json, fitted on seeds 2000-2041) to frames of unseen scene |
| `bev_rl/adaptive/diagnose_visibility_failures.py` | Replay individual FAILED episodes of the visibility-teacher prototype and log what happened in the last steps. Simulator only. Makes no assumption abo |
| `bev_rl/adaptive/brake_check.py` | How hard can the simulated ego car actually brake? (Needed before the stopping-distance rule can assume a deceleration b.) |
| `bev_rl/adaptive/visibility_teacher.py` | PROTOTYPE of the Visibility-Conditioned SafeTeacher (simulator only, no camera). New file: nothing existing is modified. |


## 3. Earlier experiments and the 200-episode study (historical; do not delete)

Named in `report/Five_Model_Comparison.pdf`, the Complete Technical Research Report (11 Nov) or the README. Several are cited by path in those reports, so moving them would break the references.

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_rl/__init__.py` | Clean PPO pipeline for the BEV driving policy (clear weather, zero traffic). |
| `bev_rl/adaptive/__init__.py` | (no docstring) |
| `bev_rl/adaptive/_smoke_degraded_pipeline.py` | SMOKE TEST ONLY (not a result): wiring of degraded_env / run_episode / AdaptPPOEnv / PPO / export_actor with a FAKE depth-error model written to a scr |
| `bev_rl/adaptive/bc_init.py` | Behaviour-clone the privileged IDM expert on the teacher's observation, to warm-start PPO (avoids the stand-still optimum). |
| `bev_rl/adaptive/bev_ablation.py` | Spec-BEV ablation: with the spec's route/boundary overlay in the BEV, what does the 6-D scalar add? |
| `bev_rl/adaptive/caution_dagger.py` | DAgger training of the caution-conditioned policy (see caution.py). |
| `bev_rl/adaptive/dagger_base.py` | Strong base policy by DAgger (policy side only). |
| `bev_rl/adaptive/degraded_dagger.py` | DAgger of a perception-degradation-aware base policy: the student sees the CALIBRATED degraded BEV (degraded_env.py), the expert is the SafeTeacher on |
| `bev_rl/adaptive/degraded_env.py` | Policy-side environment with CALIBRATED perception degradation (replaces the dropout-only proxy of caution.py). |
| `bev_rl/adaptive/diagnose_bev_fidelity.py` | Does the BEV the policy receives represent the TRUE position of the vehicle relative to the ego? Replay a model and, for the last steps before the epi |
| `bev_rl/adaptive/diagnose_scenario_fault_attribution.py` | Fault attribution for a failing scenario: who moves into whom? For the last K steps before the episode ends, log the ego's lane index and |
| `bev_rl/adaptive/diagnose_student_failure_scenario.py` | Why does a learned policy fail a scenario the teacher passes? Runs the student on one seed (density 0.05, clear, no dropout) and, at every step, |
| `bev_rl/adaptive/diagnose_teacher_crash_causes.py` | Diagnostic: for seeds where the IDM teacher crashes in traffic, log WHICH crash flag fired, ego speed, and the nearest other object |
| `bev_rl/adaptive/eval_caution.py` | Evaluation of the caution-conditioned policy on FRESH seeds (default 1010-1039; never used for training/validation). |
| `bev_rl/adaptive/eval_dagger.py` | Final evaluation of a DAgger checkpoint on the RESERVED seeds (use once per selected model) with Wilson 95% intervals. |
| `bev_rl/adaptive/eval_repo_bc.py` | Evaluate the repo's committed BC checkpoints (and the IDM expert) in the fast env under the repo's evaluation contract. |
| `bev_rl/adaptive/evaluate.py` | Evaluate teacher policies (and the IDM baseline) on reserved seeds across weather. |
| `bev_rl/adaptive/evaluate_checkpoint_sparse.py` | Evaluate ANY caution-network checkpoint (runs/<run>/iter_<k>.pt or best.pt) on the fresh sparse-traffic protocol (seeds 1010-1049, density 0.05, clear |
| `bev_rl/adaptive/evaluate_degraded.py` | Evaluation under CALIBRATED perception degradation (degraded_env.py), FRESH seeds, paired across arms. One harness for every arm. |
| `bev_rl/adaptive/evaluate_final_200seed_sparse.py` | Final large-n evaluation on a FRESH seed block (default 5000-5199; never used for training, validation or checkpoint selection). |
| `bev_rl/adaptive/evaluate_localisation_error_sensitivity.py` | Does a localisation (ego-pose-vs-HD-map) error break the policies in sparse traffic? Clear weather, density 0.05, fresh seeds, no detector dropout. |
| `bev_rl/adaptive/evaluate_pose_reliability_policy.py` | Evaluate pose-error-augmented policies (pose_reliability_dagger.py) on the SAME protocol/seeds/levels as evaluate_localisation_error_sensitivity.py |
| `bev_rl/adaptive/evaluate_pose_two_cue_policy.py` | Evaluate TWO-CUE pose-reliability policies (pose_reliability_dagger.py --cam-cue) on the localisation-error protocol (fresh seeds 1010-1049, density 0 |
| `bev_rl/adaptive/evaluate_safe_teacher_ceiling.py` | Teacher ceiling: stock IDM vs IDM+predictive safety layer, clear weather, traffic density --density, same seeds. |
| `bev_rl/adaptive/evaluate_sparse_traffic_weather_robustness.py` | Sparse-traffic (density 0.05) weather-robustness comparison on FRESH seeds, same seeds / weather / detector-dropout stream for every arm. |
| `bev_rl/adaptive/expert_ceiling_lane_change_ablation.py` | Diagnostic: what is the success ceiling of the IDM teacher in traffic, and does its lane-changing cause the crashes? |
| `bev_rl/adaptive/export_detector_table_images.py` | Regenerate and SAVE the camera pictures behind the detector retention table (retention_table.json / perception_curve.json). |
| `bev_rl/adaptive/fit_depth_error_model.py` | Fit the measured depth/detection degradation (measure_depth_error.py) into (1) a proof table + figure, (2) a sampling model for the policy-side enviro |
| `bev_rl/adaptive/measure_depth_error.py` | READ-ONLY measurement: how the GitHub front-stereo depth + YOLO detection degrade under the GitHub weather augmentation, against MetaDrive truth. |
| `bev_rl/adaptive/measure_paired.py` | READ-ONLY paired degradation curve of the GitHub YOLO+stereo->BEV pipeline under the GitHub weather augmentation. |
| `bev_rl/adaptive/measure_perception.py` | READ-ONLY measurement: does the GitHub weather augmentation degrade the GitHub YOLO+stereo -> BEV pipeline? |
| `bev_rl/adaptive/measure_retention_real_episodes.py` | READ-ONLY measurement: how often the GitHub YOLO detector still finds a vehicle in fog and rain, in REAL driving episodes at a given traffic density. |
| `bev_rl/adaptive/paired_policy_comparison.py` | Paired comparison of policies evaluated on IDENTICAL seeds (exact McNemar test + paired bootstrap CI of the success difference). |
| `bev_rl/adaptive/paired_stats_final_200seed.py` | Paired analysis of results/final_200seed/*.json (identical seeds): per-arm Wilson CI + means, pairwise exact McNemar (success) and paired bootstrap of |
| `bev_rl/adaptive/pose_reliability_dagger.py` | DAgger under random ego-pose (HD-map localisation) error, with optional RELIABILITY CONDITIONING. |
| `bev_rl/adaptive/scalar_ablation.py` | Scalar-state ablation: how does driving change when parts of the 6-D scalar state are removed? |
| `bev_rl/adaptive/vanilla_camera_cnn.py` | VANILLA camera policy: ONE front camera image -> small CNN -> (steer, throttle). Nothing else goes in (no BEV, no speed, no scalars, no weather label) |
| `bev_rl/stats.py` | Paired statistics for comparing policies on identical scenarios. |
| `bev_rl/validate_lazy_perception.py` | Prove LazyStereoPerception == FullPerception (bit-identical BEV and scalars). |
| `bev_rl/adaptive/followup_caution_sparse_traffic_queue.sh` | Sparse-traffic (density 0.05) caution study. Sequential. Nothing is deleted. Writes logs/followup_sparse_done.flag at the end. |
| `bev_rl/adaptive/queue_adaptive_night.sh` | Overnight pipeline for weather-ADAPTIVE driving under measured depth/detection degradation (see paper/depth_degradation/PREREGISTRATION.md). |
| `bev_rl/adaptive/queue_after_remeasure.sh` | Chain 2 (2026-11-10): after the CORRECTED measurement (logs/measure_v2.log ends with exit=0): refit -> sanity check -> discard invalid stage outputs - |
| `bev_rl/adaptive/queue_caution_comparison_n200.sh` | 200-episode comparison across weather of: the caution-aware model (caution derived from weather), the proposed model, and the two baseline models. |
| `bev_rl/adaptive/queue_caution_forced_one.sh` | POST-HOC control at density 0.15: caution forced to 1 in all weather, including clear. Seeds 5000-5199. Flag logs/caution_forced_one_done.flag |
| `bev_rl/adaptive/queue_caution_longer_time_limit.sh` | Caution-aware model re-evaluated with a 1600-step (160 s) limit instead of 800 steps (80 s), in the three weather cells where it timed out. |
| `bev_rl/adaptive/queue_final_200seed_then_studies.sh` | Strictly sequential (RAM ~3 GB free): no process-polling (the earlier PowerShell self-match bug deadlocked the old queues). |
| `bev_rl/adaptive/queue_n200_all_weather.sh` | n=200 all-weather evaluation of the six presented models (seeds 5000-5199, density 0.05). Strictly sequential, no process polling. |
| `bev_rl/adaptive/queue_pose_reliability_study.sh` | Waits for the safe-teacher queue (RAM), then: (1) map-only pose-reliability pair, (2) TWO-CUE pair (map overlay + noisy lane-detector scalars), (3) ev |
| `bev_rl/adaptive/queue_rerun_timeouts_1600.sh` | Re-run ONLY the episodes that hit the 800-step limit, at the 1600-step limit (the others are unaffected by a longer limit). |
| `bev_rl/adaptive/queue_retention_real_episodes.sh` | Real-episode detector retention at three traffic densities, 30 fresh scenes each (seeds 8000-8029), one burst of 2 frames every 20 steps. Flag: logs/r |
| `bev_rl/adaptive/queue_safe_student_after_weather.sh` | Waits for the n=200 all-weather queue's done-flag (file test only, no process polling), then evaluates the safe-teacher student |
| `bev_rl/adaptive/queue_safe_teacher_study_after_current_jobs.sh` | Waits until no other training/eval job is running (RAM), then: (1) teacher ceilings at sparse traffic, (2) DAgger students from the SAFE teacher. |
| `bev_rl/adaptive/queue_sparse_always_cautious.sh` | SPARSE traffic (density 0.05), seeds 5000-5199: always-cautious control (caution forced to 1, every weather incl. clear) at the 800-step and 1600-step |
| `bev_rl/adaptive/queue_weather_regime_confirm.sh` | CONFIRMATORY n=200 at density 0.15 (chosen by the pilot, see PREREGISTRATION_weather_regime.md). 16 parallel streams = 4 models x 4 weather. |
| `bev_rl/adaptive/queue_weather_regime_controls.sh` | POST-HOC controls at density 0.15, seeds 5000-5199: caution forced to 0 (caut_c0) and the model trained with caution fixed at 0 (ctrl_nocaut), 4 weath |
| `bev_rl/adaptive/queue_weather_regime_pilot.sh` | PILOT to find a traffic density where simulated weather lowers success. Fresh seed block 6000-6039 (never used before), horizon 800. |
| `bev_rl/adaptive/resume_after_reboot.sh` | Resume the adaptive-driving pipeline after a restart. Precondition: the CORRECTED measurement exists (bev_rl/adaptive/results/depth_error_measurements |
| `bev_rl/adaptive/resume_night.sh` | RESUME the adaptive-driving pipeline after a restart (2026-11-11). Stages with logs/night_<stage>.done are skipped: measure, fit and hurt are DONE and |


## 4. Reinforcement learning (handled by a teammate; what was tried is in the README, section 6)

PPO attempts 1 to 4 and their environments. **They used rescaled or redesigned rewards, not the reward in `docs/AV_Carla_BEV (1).pdf` Section 8** (comparison table in the README and in `report/latest_14th_Nov_26.pdf`). Models, logs and videos: `trained_models/04_Reinforcement_Learning_Attempts/`.

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_rl/adaptive/eval_ppo.py` | Reserved-seed evaluation of weather-aware PPO policies AND of DAgger bases, under the SAME weather x density cells. |
| `bev_rl/adaptive/ppo_adaptive.py` | PPO fine-tuning with reward v3 under CALIBRATED perception degradation, two variants from the SAME base checkpoint (so any difference is the weather i |
| `bev_rl/adaptive/ppo_bev.py` | PPO fine-tuning of the DAgger base on the repo BEV (+ 6-D scalar + explicit weather), with measured detector dropout. |
| `bev_rl/adaptive/ppo_caution.py` | PPO fine-tuning with a COMBINABLE energy + robustness reward, on the caution-conditioned BEV policy. |
| `bev_rl/adaptive/record_rl_attempt4_video.py` | Plain front-camera video of the reinforcement-learning Attempt 4 model (fine-tuned from the proposed model, no weather input), same format as record_c |
| `bev_rl/adaptive/reward_v3.py` | Reward v3 for weather-adaptive driving in sparse traffic (replaces bev_rl/reward.py for this study; the old file is untouched). |
| `bev_rl/adaptive/train.py` | PPO teacher for weather-adaptive energy-aware driving.  python -m bev_rl.adaptive.train --name C_s0 --seed 0 --mode conditioned --lambda-e 0 |
| `bev_rl/env.py` | Gymnasium wrapper: MetaDrive + the existing perception->BEV pipeline. |
| `bev_rl/evaluate.py` | Policy-agnostic closed-loop evaluation (BC and PPO share exactly this code path). |
| `bev_rl/factory.py` | Construct the perception->BEV pipeline and environments (import order matters on Windows). |
| `bev_rl/policy.py` | Actor-critic with the intended architecture, initialised from the BC checkpoint. |
| `bev_rl/reward_audit.py` | Reward sanity audit: scripted behaviours must rank sensibly under the step reward. |
| `bev_rl/adaptive/followup_queue_after_ppo_sweep.sh` | Runs AFTER rl_queue.sh finishes (RAM: one heavy job at a time). Order: caution DAgger + control, then fresh-seed evals. |
| `bev_rl/adaptive/rl_queue.sh` | Sequential PPO reward-combination sweep (RAM-limited: one combo at a time, 2 workers). Start = ov_split + caution input. |


## 5. Videos and recording helpers

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_rl/adaptive/record_camera_plain.py` | PLAIN front-camera videos in the format of bev_policy/outputs/videos/fog_seed1000.mp4 (teammates' reference): |
| `bev_rl/adaptive/record_front_camera.py` | FRONT-CAMERA recordings of every model on ONE seed under different weather (clear / fog 0.5 / rain 0.5 / ...). |
| `bev_rl/adaptive/record_pixel_policy_video.py` | Record the IMAGE-ONLY driving policy (vcam) driving, as a front-camera video in the same format and resolution as the other report videos. |
| `bev_rl/adaptive/record_videos.py` | Record a video of every model driving the SAME scenario (paired seeds). Policy-side tool; read-only on checkpoints. |
| `bev_rl/adaptive/record_all_camera.sh` | Front-camera recordings, ONE seed (1039: oncoming vehicle drifts into the ego lane), every model, clear / fog 0.5 / rain 0.5.  Resumable (--skip-exist |
| `bev_rl/adaptive/record_all_front_camera.sh` | PLAIN front-camera videos (format of bev_policy/outputs/videos/fog_seed1000.mp4), ONE seed (1039), every model, fog 0.5 / rain 0.5 / clear, then fog 1 |
| `bev_rl/adaptive/record_all_videos.sh` | Record every model on paired seeds -> paper/videos/<model>__seed<N>__<weather>.mp4 (+ INDEX.md). Resumable (--skip-existing). Log: logs/videos.log, fl |


## 6. Superseded (do not use)

`e2e_dagger.py` used SafeTeacher for the OV-Split recipe, which is wrong (the original used the stock IDM), and shared rollouts between the two models; it was pushed in commit `4d45920` and never produced data. `e2e_dagger_both.py` is the local variant, never run. `queue_e2e*.sh` and `evaluate_e2e.py` belong to the superseded fine-tuning plan; they were replaced by `camera_dagger.py` and `evaluate_camera_pipeline.py`. `e2e_pipeline.py` itself is **still used** (base class `E2EEnv`).

| File | What it is (first line of its own docstring or header) |
|---|---|
| `bev_rl/adaptive/e2e_dagger.py` | Fine-tune the proposed model on grids built by the FULL camera pipeline (dataset aggregation through cameras -> weather filter -> stereo depth + YOLO  |
| `bev_rl/adaptive/e2e_dagger_both.py` | Fine-tune BOTH existing policies (proposed ov_split and caution-aware caut_a) on grids from the full camera pipeline + CameraGridBuilder. |
| `bev_rl/adaptive/evaluate_e2e.py` | END-TO-END evaluation: the policy's grid comes from the cameras (weather filter -> stereo depth + YOLO -> footprints -> grid), see e2e_pipeline.py. |
| `bev_rl/adaptive/queue_e2e.sh` | FULL END-TO-END evaluation (cameras -> weather filter -> stereo depth + YOLO -> grid -> policy). 3 workers pull from one prioritised queue; results pe |
| `bev_rl/adaptive/queue_e2e_finetune.sh` | FULL-PIPELINE (camera) plan, run unattended, in this order: |


## 7. Not ours (teammates' modules; do not modify)

`camera/`, `YOLO+weather module/`, `bev_state/assembler.py`, `bev_policy/` (baselines; three files carry local backwards-compatible edits), `baseline/bc_metadrive/`.

## 8. Proposed cleanup (awaiting approval; nothing done)

1. Move the 5 superseded files in section 6 to `archive/superseded_2026-11-14/` with an index of original paths (reversible, no deletion). Safe now: none is imported by active code or hashed in the manifest.
2. Add a one-line "status: historical" docstring header to older files that have none (no code change).
3. After OV-Split finishes: write the Live Caution trainer as a new file and remove the retired `caut` sensor branch from `camera_dagger.py`.
4. Do **not** reorganise queue scripts or the historical study scripts: the running orchestrator calls scripts by path and the reports cite them by path.
5. Before any push, review what the new untracked folders contain (`.gitignore` already excludes `*.npz`, videos and weights).
