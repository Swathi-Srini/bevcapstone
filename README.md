# BEV Autonomous Driving Research Project

> **Maintained project source of truth.** Update this file whenever the architecture, implemented status, research claim, or project milestone changes. Module READMEs explain only how to use their own code.
>
> **Phase 3 update (17 November 2026):** the image-based severity-adaptive system, its 20-seed results and the paper draft are described in [`docs/phase3/README.md`](docs/phase3/README.md). The sections below are the 14 November state.
>
> **Last updated: 14 November 2026, evening.** Full technical record of everything up to this date: [`report/latest_14th_Nov_26.pdf`](report/latest_14th_Nov_26.pdf). Methodology and schedule only: [`report/Final_Methodology_and_Approach.pdf`](report/Final_Methodology_and_Approach.pdf).

**Project title: *Reinforcement learning under fog and rain conditions*.** Current work: proper **base models** (OV-Split Camera, Live Caution Camera) trained on our own camera pipeline; the reinforcement-learning part is handled separately by a teammate (what was tried: section 6; code map: [`docs/REPO_MAP.md`](docs/REPO_MAP.md)). A MetaDrive-based research project on **driving under simulated fog and rain** with a camera-derived Bird's-Eye-View (BEV) policy. **The final simulator is MetaDrive, not CARLA** (earlier CARLA work is design context only).

## 1. Where we are (14 November 2026)

| Item | Status |
|---|---|
| Camera pipeline (5 cameras → synthetic weather → YOLOv8n + front StereoSGBM depth → assembler) | teammates' modules, unchanged |
| **Camera grid builder v2** (`bev_state/camera_grid_builder.py`) | ours; implemented, calibrated, **frozen** by a SHA-256 manifest |
| Pilot of the full pipeline (original OV-Split checkpoint, 40 episodes) | done; results in section 5 |
| Teacher design and validation (simulator only) | done (Teacher A, Teacher B) |
| **OV-Split Camera** (DAgger through the full camera pipeline) | **running**: rounds 0 and 1 complete and retrained (`iter_0.pt`, `iter_1.pt`), round 2 collecting |
| Retention measurement on the full pipeline → new caution values | **not run yet** (design approved in principle; exact configuration pending) |
| **Live Caution Camera** | **not started** (trainer with table-derived caution not yet written) |
| Final camera-pipeline evaluation (10 matched seeds × 4 weathers) | code written, **never run**, queued |
| Original ("200-episode") models on the camera pipeline | queued (fog 1.0 × 10 episodes + videos) |
| Paper, report, conference selection | in progress, protected time 17–19 Nov |

Hard deadlines: essential experiments by 16 Nov; paper/report/PPT 17–19 Nov; guide approval and submission preparation 20–22 Nov; project review 23 Nov.

## 2. Naming guide (the names get confusing)

| Name | Meaning |
|---|---|
| **200-episode models / study** | The original checkpoints (`runs/ov_split/best.pt`, `runs/caut_a/best.pt`) trained on **simulator ground truth**, and the 200-episode, five-model study (seeds 5000–5199). That study used the simulator's vehicle list with a measured dropout, **not** the cameras. |
| **Camera students** | The two models trained now through the full camera pipeline: **OV-Split Camera** (non-adaptive) and **Live Caution Camera** (adaptive). |
| Teacher A / Teacher B | SafeTeacher + curve guard 4.0 (clear) / caution IDM + SafeTeacher + curve guard 4.0 (caution). |
| Retention-conditioned caution | `c = clip(1 − retention(weather), 0, 1)`, retention measured on the full pipeline. |

## 3. What the system is

Every simulation step (10 Hz):

```text
5 RGB cameras (1200x900, 60 deg FOV)
  -> synthetic weather filter on every picture            (YOLO+weather module/weather/weather_utils.py)
  -> front StereoSGBM depth  +  YOLOv8n on 4 views         (stereo Z = f*B/d, f = 1039.23 px, B = 0.5 m)
  -> detection -> ego position (stereo for front, ground-plane for side/rear)
  -> camera grid builder v2 (our module) -> 64x64 BEV (+ HD-map layers)
  -> policy: BEV CNN + 3 ego scalars (+ caution scalar for Live Caution)
  -> action [steering, throttle/brake]
```

No simulator vehicle position reaches the policy. Ground truth is used only by the privileged teacher (to label states) and offline to score perception and driving. Ego scalars and the HD-map layers come from the simulator's map and ego state, as in the Review 1/2 architecture.

**Weather module (read from the code):** an image-space filter with no depth dependence. Fog = white blend rising linearly from the top row (0) to the bottom row (0.9·intensity), Gaussian blur, noise. Rain = 800·intensity random streaks, blend, and a vertical filter whose weights sum to 0.75, so **every rain picture is darkened 25 %**. Weather never changes grip or physics. The two front pictures get the identical random draw (stereo needs it).

**Network** (`bev_policy/model.py`, `BEVScalarPolicy`): BEV CNN (32/64/96 maps) + scalar MLP 6→64→32 → fusion 160D → head 160→128→2 (tanh), 110 530 parameters. Live Caution adds a caution branch FC(1→16): fusion 176D, 112 610 parameters. The scalar vector is 6D (`v, progress, d_lat, θ_err, κ, d_goal`) but a fixed mask `[1,1,0,0,0,1]` zeroes three of them (the BEV overlay carries them), so 3 are effective.

## 4. Key findings so far

1. **The 200-episode study did not run the full pipeline** (it used the simulator vehicle list with a dropout table). Discovered 14 Nov; the grids never matched on frames containing a vehicle. The results are kept as a *labelled ideal-perception reference*.
2. **The camera pipeline has an integration issue:** `left_camera`/`right_camera` are point-reflected relative to the assembler's position conversion; stereo reports the nearest surface, not the centre; YOLO misses large side-on vehicles at 4–7 m and edge-truncated ones; recall collapses in rain; the five 60° cones leave blind wedges.
3. **Builder v2** corrects these inside our module (side reflection, fitted affine centre correction, true-size footprint 2.0 × 4.9 m, duplicate merge). Calibrated on 85 point pairs (seeds 2000–2002); leave-one-scene-out median error front 1.58 → 0.91 m, side 11.4 → 1.3 m.
4. **Exact grid equality on vehicle frames is not achievable** (0/82, 0/180, 0/216, 0/177 in clear/fog/rain 0.6/rain 1.0) because of detector limits; empty-window frames match 99.7–100 %.
5. **The old retention table was YOLO box counts** (`measure_paired.py`, 20 scenes, stationary ego, ≤3 vehicles at ~8–14 m), not full-pipeline reliability. Fog 1.0: table keeps 27 %; the full chain keeps about 4 % of clear-weather recall. It is retired.
6. **The always-cautious control matched the adaptive caution model** on success and energy in weather; the adaptive model's real advantage is **trip time in clear weather** (43.5 s vs 64.2 s).
7. **Teachers:** SafeTeacher + curve guard 4.0 reaches 0.99 success in clear (seeds 3100–3199); the caution teacher reaches 0.98–1.00 in weather without "safe by stopping".

## 5. Results

### 5.1 Pilot of the full camera pipeline (original OV-Split checkpoint; builder v2; seeds 5000–5009; density 0.05; 800 steps)

| | Clear | Fog 1.0 | Rain 0.6 | Rain 1.0 |
|---|---|---|---|---|
| Success / collisions / off-road / timeout | 9/1/0/0 | 8/2/0/0 | 8/2/0/0 | 8/2/0/0 |
| Completion s / speed m/s / Wh/km | 44.0 / 7.77 / 97.6 | 40.4 / 7.94 / 94.3 | 40.4 / 7.90 / 101.6 | 40.4 / 7.94 / 93.1 |
| Vehicle instances in window (frames with a vehicle) | 97 (82) | 421 (180) | 500 (216) | 420 (177) |
| YOLO detections on those frames | 88 | 10 | 50 | 3 |
| Recall within 3 m: original builder → **v2** | 0.351 → **0.474** | 0.002 → **0.019** | 0.080 → **0.082** | 0.000 → **0.000** |
| Median position error original → v2 (m) | 1.41 → 0.93 | 0.84 → 0.49 | 2.08 → 1.09 | – |
| Duplicates original → v2 | 6 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |

Same seeds on the simulator-list grid: 9, 9, 8, 8 successes; **33 of 40 episodes identical** in outcome and step count. Preliminary retention from this pilot: fog 1.0 ≈ 0.04, rain 0.6 ≈ 0.17, rain 1.0 = 0 (relative to clear 0.474) — *not for use* until the measurement and review described in section 7 are done.

### 5.2 The 200-episode, five-model study (simulator-list grid + measured dropout; seeds 5000–5199; density 0.05; 800 steps)

Models: (1) teammates' baseline trained without traffic, (2) teammates' baseline trained with traffic (both 5-demo behaviour cloning, 110 530 params, no road drawing), (3) **Proposed** BEV model (`ov_split`: road drawn into the grid, 3 scalars, DAgger rounds 0–5, 73 032 labelled steps, stock-IDM teacher), (4) **Caution-aware** model (`caut_a`; +caution branch; caution teacher 30(1−c/3) km/h and gap 10+15c m; best validation round 1, 35 503 steps), (5) **Image-only** CNN on the real front camera (no grid, no speed; 37 692 pictures; clean pictures only).

Success rate (80 s limit):

| Model | Clear | Fog 1.0 | Rain 0.6 | Rain 1.0 |
|---|---|---|---|---|
| 1 Baseline, no traffic | 0.850 | 0.860 | 0.850 | 0.850 |
| 2 Baseline, with traffic | 0.745 | 0.750 | 0.755 | 0.755 |
| 3 Proposed | 0.925 | 0.895 | 0.900 | 0.900 |
| 4 Caution-aware | 0.890 | 0.895 | 0.890 | 0.820 |
| 5 Image-only | 0.130 | 0.000 | 0.030 | 0.010 |

Speed / energy (Wh/km): models 1–3 about 8 m/s and 86–96 Wh/km; model 4 in fog/rain 5.6–6.2 m/s and 59.7–62.7 Wh/km (about 35 % lower); model 5 about 9 m/s and 277–392 Wh/km (mostly leaves the road).

160 s limit (post-hoc, set after seeing the time-outs): model 4 = 0.940 / 0.955 / 0.935 in fog 1.0 / rain 0.6 / rain 1.0 vs model 3 = 0.895 / 0.900 / 0.900; paired p = 0.136 / 0.061 / 0.296 (not significant).

**Always-cautious control** (caution forced to 1 even in clear): 0.930 / 0.935 / 0.935 / 0.945, speed 5.5 m/s, 59 Wh/km in every condition, 64 s trips — as good as model 4 in weather, and 48 % slower in clear. So the gain is cautious driving, not weather reading.

**Busy traffic (density 0.15, 200 episodes):** model 3: 0.515 (clear) → 0.425 / 0.380 / 0.370 (fog 1.0 / rain 0.6 / rain 1.0); model 4: 0.355 (c = 0 in clear) → 0.525 / 0.565 / 0.580; model 4 with c forced to 1: 0.565–0.590; baselines 0.26–0.34.

Paired tests (exact McNemar): model 3 vs model 1 +0.075 (p = 0.0007), vs model 2 +0.180 (p < 0.0001); everything vs image-only p < 0.0001.

**What it supports / does not support:** supports "BEV with road drawing + DAgger ≫ 5-demo BC" and "caution-aware behaviour slows down when caution is high, ~35 % less estimated energy, fewer collisions (2.5–5 % vs 8.5–9.5 %)". Does **not** support "senses weather" (caution is supplied), "weather-specific gain" (always-cautious control), "more efficient, not just slower" (no speed-matched control), or anything about a real detector/camera.

Original-checkpoint 10-episode slice (seeds 5000–5009, same simulator-grid setting): OV-Split 9/9/8/8 successes, caution-aware 8/9/9/9 (clear/fog/rain 0.6/rain 1.0); caution-aware speed 5.6–6.1 m/s and 57–69 Wh/km in weather vs 7.8–7.9 m/s and 96–106 Wh/km for OV-Split.

### 5.3 Teacher validation (simulator only, seeds 3100–3199, density 0.05, 160 s)

| Teacher | Weather | Success@160 | Crash | Off-road | Speed m/s | Completion s | Wh/km |
|---|---|---|---|---|---|---|---|
| Stock IDM | clear | 0.95 | 0.02 | 0.03 | 7.96 | 43.7 | 88.8 |
| SafeTeacher | clear | 0.97 | 0.00 | 0.03 | 7.90 | 44.3 | 92.6 |
| **A: SafeTeacher + guard 4.0** | clear | **0.99** | 0.00 | 0.01 | 7.80 | 45.0 | 99.0 |
| **B: caution IDM + SafeTeacher + guard 4.0** (old c) | fog 1.0 / rain 0.6 / rain 1.0 | 0.98 / 1.00 / 1.00 | 0.01 / 0 / 0 | 0.01 / 0 / 0 | 6.09 / 5.82 / 5.52 | 57.6 / 60.2 / 63.5 | 67.1 / 63.8 / 60.2 |

Anti-stopping gates held (stopped ≤ 0.54 % of steps, hard-brake ≤ 0.15 %). Congestion check at density 0.1 (50 seeds): stock IDM 0.84 (12 % crashes), A 0.96, B 0.96 / 0.96 / 0.94. Teacher B must be re-validated at the new caution values before use.

## 6. Reinforcement learning (what was tried; part of the plan, not the current focus)

PPO in all four attempts, 2-D action `[steering, throttle/brake]`, starting from an imitation-trained network (a random start settles at standing still). None beat the imitation model it started from — **with rescaled or redesigned rewards, not the reward in the team specification** (see the comparison below).

| # | Attempt | Result | Lesson |
|---|---|---|---|
| 1 | Privileged-state teacher, small network, reward v1, speed cap 10/12, crash penalty 50, rain lowers tyre grip (later removed); runs C2, Probe C, C3 (1.5 M / 1 M / 0.3 M steps) | 16/70 succeed (stock driver 47/70); all failures leave the road | a rising reward is not better driving |
| 2 | Fine-tune the proposed model with a weather input, 8 weather mixes with vehicle loss, density 0.1 | 0.675 / 0.65 / 0.675 vs start: no change | a weather input cannot help a model that weather does not hurt |
| 3 | Five reward combinations (energy term −0.05·E, caution-dependent speed limit and crash penalty, headway) | 0.325–0.400 vs start 0.350: noise; energy did not fall | 40 episodes cannot select a model; a reward cannot add missing information |
| 4 | Reward v3 (time-to-collision and gap terms, no brake penalty, energy −0.02·E, +50 / −100 terminal) on the weather-degradation model; 200 test episodes | success **lower** by 0.065, 0.115, 0.115, 0.095 (significant): faster, less safe | reward changes alone reduced safety |

Reward v1 per step: `min(s, vcap)/(vref·dt) − 0.5 d² − 5 ψ² − 0.1‖a_t − a_{t−1}‖² − 0.002 j² − 0.2 max(0, −a_thr) − 0.5 ((v − vcap)+)² − 0.2·[v<0.5] + R_end` (v1: +20 / −50). Models, logs and videos are under `trained_models/04_Reinforcement_Learning_Attempts/`.

### Reward actually used vs the intended reward (spec: `docs/AV_Carla_BEV (1).pdf`, Section 8)

The intended reward is `R_t = r_progress + r_speed + r_energy + r_lateral + r_heading + r_brake + r_jerk + r_still + r_speed_cap + r_lane` plus terminal rewards (goal +500, collision −200, lane violation −30·|d|, timeout 0), written for CARLA at 0.04 s steps and three outputs [steering, throttle, brake]. **None of the four attempts used these weights or this form.** (The spec's weight table is garbled in text extraction; the value-to-term pairing below follows the table order.)

| Term | Intended (spec) | Attempts 1–3 (`bev_rl/reward.py`, "v1") | Attempt 4 (`bev_rl/adaptive/reward_v3.py`) |
|---|---|---|---|
| Progress | 10·Δs | 1.0·Δs/(v_ref·dt), credit capped at the speed cap | same |
| Speed × lane-OK | 1.0·v·max(0, 1−\|d\|/3) | **absent** | **absent** |
| Energy | −0.01·E | **absent** in the base run; sweep variant −0.05·Wh | −0.02·Wh |
| Lateral | −3.0·d² | −0.5·d² | −0.5·d² |
| Heading | −8.0·ψ² | −5.0·ψ² | −5.0·ψ² |
| Brake | −0.5·brake | −0.2·max(0, −throttle) | **removed on purpose** |
| Jerk | −0.1·(Δa)² | −0.002·j² (physical jerk) + an extra action-change term −0.1·‖Δa‖² (not in spec) | same |
| Still penalty | −5.0 if v<0.5 and no progress | −0.2 if v<0.5 | −0.2 if v<0.5 |
| Speed cap | −0.5·max(0, v−12)² | −0.5·((v−cap)/2)², cap 12 (C2, Probe C) or 10 (C3) | **absent** (progress credit capped at 10 m/s only) |
| Lane violation | −30·max(0, \|d\|−3) per step | **absent** | **absent** |
| Terminal goal / collision / lane | +500 / −200 / −30·\|d\| | +20 / −50 (PPO runs; code default −10) | +50 / −100 |
| Added, not in spec | – | – | time-to-collision term −2·(1−τ/3)² and gap term −1·(1−g/6)² |

Consequences: the spec's weights were never tried (the first reward file's docstring says its weights are re-scaled and "NOT the PDF's CARLA weights"); the spec's terminal magnitudes are 10–25× larger and its collision-to-success ratio (−200/+500 = 0.4) is inverted in our runs (−50/+20 = 2.5 in v1, −100/+50 = 2.0 in v3); the speed × lane-OK reward and the per-step lane-violation penalty were never present; MetaDrive has two outputs, so the brake term became a negative-throttle term; the spec's Phase 2 (RL without traffic plus energy) and Phase 3 (traffic plus domain randomisation, visibility-based control) were not followed (attempts 2–4 used sparse traffic and weather as vehicle dropout). **Allowed wording:** "reinforcement learning with our rescaled and redesigned rewards did not improve on imitation"; not "reinforcement learning with the intended reward fails" (untested). The energy accounting also differs (spec: engine-force power model for a 1800 kg CARLA vehicle at 25 Hz; ours: closed-form per-step mechanical work with the simulator's vehicle mass, no gradient).

## 7. Final methodology (condensed; full text in `report/Final_Methodology_and_Approach.pdf`)

**Retention-conditioned caution.** `weather setting → measured retention r → c = clip(1 − r, 0, 1) → teacher and student`. Retention is measured on the *full pipeline* against simulator ground truth: `recall(w) = (vehicles in the BEV window with a corrected object within 3.0 m) / (vehicles in the window)`, `r(w) = min(1, recall(w)/recall(clear))`, cluster-bootstrapped over episodes. Hybrid measurement: pilot data for clear/fog 1.0/rain 0.6/rain 1.0 plus a new 20-episode pass (fog 0.6, rain 0.2, rain 0.45 and 5 clear episodes as a denominator check); pilot and new data are labelled separately, denominators are checked, and the final values are reviewed before use. The weather is supplied, so this is **not** image-only adaptation; it is constant within an episode in the approved collection.

**Teachers.** IDM (lane change off) + SafeTeacher (constant-velocity time-to-collision over 3 s; full brake below 1.3 s, graded braking below 2.4 s) + curve guard (`v_cap = max(3.5, sqrt(4.0/κ_max))`). Teacher B adds `v0(c) = 30(1 − c/3) km/h`, gap `10 + 15c` m.

**DAgger through the camera pipeline** (both models): density 0.05 only; 40 episodes per round; rounds 0–5 (aim), with a reduced-round fallback (stop after 4 completed rounds, retrain on all data, validate); every step through the full pipeline and recorded; `β = 1` then `0.5^t`; action noise 0.1; horizon 600; Adam 1e-3, batch 128, MSE; epochs 24/8 (OV-Split), 18/6 (Live Caution); initialised from the original checkpoints; training seeds from 2000–2999; validation seeds 3100–3109 at density 0.05, horizon 800. OV-Split trains in clear weather only; Live Caution uses the original 8-item weather mix and the table-derived `c` for teacher and student.

**Evaluation** (final): seeds 5100–5109 × {clear, fog 1.0, rain 0.6, rain 1.0}, density 0.05, 800 steps, no noise, full pipeline, frozen builder; Wilson intervals, exact McNemar; energy (formula in the PDFs); perception metrics against ground truth; front-camera + BEV videos (seed 5100: clear, fog 1.0, rain 1.0). Time-outs at 800 steps are re-run at 1600 and reported separately.

**Seed registry.** 2000–2002 builder calibration; 2000–2041 old sensor fit (archived); 2000–2999 DAgger training; 3100–3199 teacher validation (3100–3109 DAgger validation); 5000–5009 pilot; 5000–5199 original 200-episode study; **5100–5109 final camera evaluation** (never used for calibration, fitting, selection or training); 6000–6009 reserved.

**Integrity.** `results/camera_dagger_final/FROZEN_MANIFEST.json` hashes the builder, assembler, the original checkpoints, YOLO weights and the old table; every training/evaluation worker refuses to run if any hash differs. New data and checkpoints go to new folders; nothing is deleted or overwritten.

## 8. Training progress (OV-Split Camera)

| Round | Episodes | Samples | Outcomes | Retraining |
|---|---|---|---|---|
| 0 (teacher drives) | 40 | 18 419 | 39 success, 1 timeout | 24 epochs, 5.2 min, loss 0.0044 → `iter_0.pt` |
| 1 | 40 | 18 131 | 37 success, 3 timeout | on 36 550 samples, 3.5 min, loss 0.0266 → `iter_1.pt` |
| 2 | in progress | | | |

Measured speed (3 workers): 1.96 s/step, ~460 steps and 15.0 min per episode per worker, 12 episodes/hour, 3.4 h per round. Each worker holds ~5.2 GB committed memory (13.7 GB machine, integrated GPU shares RAM); 3 workers is the proven maximum, a 4th is tested at the start of the queued evaluation pack. Expected OV-Split completion 15 Nov ~16:00 (earliest 13:30, latest 18:30).

## 9. Queue (local-only file `logs/QUEUE.md`; `logs/` is git-ignored)

1. OV-Split Camera DAgger rounds 0–5 — **running**.
2. `queue_pack.sh` — after OV-Split: original OV-Split and caution models on the full camera pipeline, fog 1.0 × 10 episodes (seeds 5100–5109) + videos (seed 5100: clear, fog 1.0, rain 1.0), with a 4-worker health test.
3. `queue_students_eval.sh` — camera-trained OV-Split student, same set.
4. Not queued yet (awaiting approval of exact configurations): retention pass, table review, targeted teacher re-validation, Live Caution DAgger, final evaluations.
5. `queue_camera_eval.sh` — deferred: remaining original-model weathers, after Live Caution.

## 10. Where things are

| What | Where |
|---|---|
| Camera grid builder v2 (frozen) | `bev_state/camera_grid_builder.py` |
| Camera / YOLO / weather / stereo modules (teammates') | `camera/`, `YOLO+weather module/`, `bev_state/assembler.py` |
| Camera-pipeline DAgger trainer (OV-Split) and orchestrator | `bev_rl/adaptive/camera_dagger.py`, `queue_camera_training.sh` |
| Teacher definitions / SafeTeacher | `bev_rl/adaptive/safe_teacher.py`, `teacher_eval.py` |
| Pilot, grid comparison, calibration | `pilot_camera_builder.py`, `compare_grids.py`, `calibrate_camera_geometry.py` |
| Camera evaluation, summary, queues | `evaluate_camera_pipeline.py`, `summarise_camera_eval.py`, `queue_pack.sh`, `queue_students_eval.sh`, `queue_camera_eval.sh` |
| Archived exploratory work | `visibility_sensor.py`, `visibility_teacher.py`, `brake_check.py`, `diagnose_visibility_failures.py` |
| Results | `bev_rl/adaptive/results/` — `n200_all_weather/` and `n200_horizon1600/` (200-episode study), `regime_confirm/` (busy traffic), `sparse_always_cautious/`, `pilot_camera_builder_v2/`, `grid_comparison/`, `camera_geometry/`, `teacher_validation/`, `teacher_density/`, `visibility_sensor*/`, `visibility_teacher_proto/`, `camera_dagger_final/` (running), `camera_dagger/` (archived first attempt, unused) |
| Original checkpoints | `bev_rl/adaptive/runs/ov_split/best.pt`, `runs/caut_a/best.pt` (never overwritten); all final models also in `trained_models/` |
| Documents | `report/` (PDFs and LaTeX): `latest_14th_Nov_26.pdf`, `Final_Methodology_and_Approach.pdf`, `Five_Model_Comparison.pdf`, `Effective_Range_Measurement_Protocol.pdf`, `Live_Caution_Sensor_and_Episode_Walkthrough.pdf`, `GitHub_vs_This_Work.pdf` (partly stale). `Directors_Research_Guide.pdf` and `Complete_Technical_Research_Report.pdf` are **local-only** (`report/local_only/`, git-ignored; not on GitHub): ask the maintainer or see `docs/BACKUP_STRATEGY.md`. What is tracked versus local: `report/README.md` |
| **Code map** (which files are used / historical / RL / superseded; how training uses `assembler.py`) | [`docs/REPO_MAP.md`](docs/REPO_MAP.md) |
| **Function-level map** (per-step call chain; what each file/function does, why, who calls it) | [`docs/FUNCTION_MAP.md`](docs/FUNCTION_MAP.md) |
| **Geometric corrections** (side reflection, affine centre fit, true-size footprint, duplicate merge; where each is applied; calibration data; reproducibility audit) | [`docs/GEOMETRIC_CORRECTIONS.md`](docs/GEOMETRIC_CORRECTIONS.md) |
| **Status of every script** (`active / done / historical / superseded / untested / rl-handoff`) | [`docs/STATUS_INDEX.md`](docs/STATUS_INDEX.md) |
| **Git workflow** (branch roles; do not run git in a working copy with running jobs; line endings) | [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) |
| **Backups and checkpoint integrity** | [`docs/BACKUP_STRATEGY.md`](docs/BACKUP_STRATEGY.md), [`docs/CLEANUP_LOG.md`](docs/CLEANUP_LOG.md) |
| **Audit of files that are not needed** (nothing deleted; archive candidates only) | [`docs/UNNEEDED_FILES_AUDIT.md`](docs/UNNEEDED_FILES_AUDIT.md) |
| Project history | `docs/project_history/` (1–5 Nov notes; may cite files that later moved, see its README), `archive/` |
| Logs and queue state | `logs/` (`QUEUE.md`, `camera_training_ov.log`, worker logs) |

**Branches:** `main` is the integration branch and contains the camera-pipeline work (PR #14, merge `7e31b6f`). `cleaned-latest` (`4d45920`) is a frozen snapshot of the 13 Nov work and is kept unchanged. `4d45920` includes `e2e_dagger.py`, which used the wrong teacher; it is superseded and never produced data. Cleanup changes go through pull requests from a separate clone (see `docs/GIT_WORKFLOW.md`).

## 11. Limitations (state these in any write-up)

Synthetic, non-physical weather filter; simulation only; the weather setting is supplied to the caution mechanism; ten episodes per condition in the camera evaluation (wide intervals); sparse traffic (few vehicles matter); privileged teacher with design-choice constants; the builder's corrections compensate for an integration issue and were fitted on three scenes (rear-camera correction unvalidated); retention is position-based recall with a 3.0 m criterion; the camera-evaluation code is untested until it first runs.

---

# Original project description and instructions (still valid)

This repository began as a vision-derived Bird's-Eye View (BEV) controller trained with behavioural cloning (BC) and a reproducible sparse-traffic baseline. Since then it has gained: DAgger-trained BEV models, a caution-aware model, a 200-episode five-model study, four reinforcement-learning attempts (none better than imitation), a camera grid builder, a full camera-pipeline pilot, teacher design and validation, and the camera-pipeline training now running. Sections 1-11 above are current; everything below the line is the original module-level description and run instructions, which remain valid.

**The final simulator is MetaDrive, not CARLA.** Earlier CARLA work is retained as exploratory/design context; it is not the implementation target.

## Start here (added 11 November 2026)

Two documents describe the project (LaTeX sources and PDFs; rebuild with `powershell -File report/build_documents.ps1`). **They are local-only** (`report/local_only/`, git-ignored), so they are not on GitHub:
- `Directors_Research_Guide.pdf` (about 20 pages): what was built, tried, found and what to do next, in accessible language.
- `Complete_Technical_Research_Report.pdf` (about 85 pages): every model, architecture, training strategy, result, reinforcement-learning attempt, failure analysis, comparison with this GitHub repository, paper directions, related 2025--2026 work and an inventory of every file (as of 11 Nov).

All final trained models are in one folder, [`trained_models/`](trained_models/), with descriptive file names, a manifest of hashes (`MODEL_MANIFEST.json`, verified: 33 of 33 hashes match) and a mapping from the previous file names (`MODEL_NAME_MAPPING.md`). The `trained_models/` copies are **tracked in git**, including the two original checkpoints (`Proposed_BEV_Driving_Model.pt`, `Caution_Study_Caution_Aware_Driving_Model.pt`), which are byte-identical to `bev_rl/adaptive/runs/ov_split/best.pt` and `runs/caut_a/best.pt`. The `runs/` folders, datasets (`*.npz`), videos and logs are git-ignored and local to the maintainer's machine (backup policy: `docs/BACKUP_STRATEGY.md`).

## Repository layout

```text
bevcapstone/
|- report/                   # Documents (LaTeX + PDF): latest_14th_Nov_26, Final_Methodology_and_Approach, Five_Model_Comparison, protocols, directors guide, technical report
|- trained_models/           # All final trained models (descriptive names) + MODEL_MANIFEST.json + Training_Records/
|- baseline/bc_metadrive/    # Preserved 259-D vector-observation BC baseline
|- camera/                   # Five-camera MetaDrive capture + front SGM depth
|- YOLO+weather module/      # Live YOLO, synthetic weather (weather/weather_utils.py), stereo depth, camera rig (teammates')
|- bev_state/                # BEV assembler (teammates') + camera_grid_builder.py (OURS, frozen)
|- bev_policy/               # BEVScalarPolicy, BC training, evaluation (teammates' baselines; three files carry local edits)
|- bev_rl/                   # Policy-side work
|  |- adaptive/              # Teachers, DAgger (simulator-grid and camera pipeline), 200-episode evaluation, pilot, calibration, queues
|  |  |- results/            # All result folders (n200_*, regime_confirm, pilot_camera_builder_v2, teacher_validation, camera_dagger_final, ...)
|  |  `- runs/               # Training runs; ov_split/ and caut_a/ are the ORIGINAL checkpoints (never overwritten)
|- repository_tools/         # Scripts that organise the repository
|- docs/                     # Specification PDF; project_history/ = dated notes 1-5 Nov 2026
|- archive/                  # Superseded material kept for the record
|- logs/  tests/  weights/   # Run logs and queue state (logs/QUEUE.md), unit tests, third-party YOLO weights
|- venv/                     # Local Python environment (ignored by Git)
|- requirements.txt          # Shared Python dependencies, including MetaDrive 0.4.3
`- .gitignore
```

## Documentation policy

There is one project-level document: this root README. It owns the architecture,
research scope, system status, and experiment plan. Each executable module has
one local README containing its commands, inputs, outputs, and limitations:

- baseline/bc_metadrive/README.md — preserved vector BC baseline;
- camera/README.md — five-camera capture and depth export;
- YOLO+weather module/README.md — live perception demonstrator;
- YOLO+weather module/stereo_depth/README.md — reusable stereo utilities.
- bev_state/README.md — BEV/state contract and live validation visualiser;
- bev_policy/README.md — BEV behavioural-cloning commands and limitations.

Do not overwrite `baseline/bc_metadrive/` during BEV development. It is the comparison experiment. Future BEV code should live separately, for example in a top-level `bev_policy/` folder.

## Target architecture

```text
MetaDrive cameras + ego/route state
  -> YOLO detections + depth/projection + lane/route geometry
  -> 64 x 64 BEV grid + 6-D scalar state
  -> CNN branch (BEV) + MLP branch (scalars)
  -> feature fusion + two-output policy
  -> MetaDrive action [steering, throttle/brake]
  -> MetaDrive environment, reward, and next observation
```

The target camera rig uses five physical streams: a front-left/front-right
stereo pair plus left, right, and rear cameras. The rig, frame capture,
front StereoSGBM depth, and a live YOLO/depth visualisation are implemented as separate prototypes. The integrated BEV builder, scalar-state extractor, CNN/MLP policy, clear-condition BC collection/training, and closed-loop evaluation are implemented. Route/lane/boundary overlays, DAgger training with traffic, energy accounting and a caution-aware model are implemented in `bev_rl/` (see sections 1-11 above). PPO was tried and did not beat imitation. Validated *image-derived* visibility adaptation is **not** implemented; the current plan is retention-conditioned caution with the weather supplied.

### Rules for future implementations

- Use MetaDrive's two-value action contract: `[steering, throttle/brake]`.
- Do not feed exact simulator NPC positions/sizes to the learned policy. They are debugging/evaluation ground truth only.
- Route/lane geometry and ego localisation are allowed structured inputs.
- `free` and `unknown` must be distinct BEV states; visibility-aware BEV needs an explicit unknown/knownness representation.
- The YOLO/weather runner uses clean RGB by default. Its
  `--perception-weather` mode applies synthetic image corruption to YOLO and
  stereo, but this is not physical weather simulation.
- On Windows CPU laptops, favour headless/small-camera debugging. MetaDrive onscreen rendering can exhaust shared graphics memory.

## Environments and dependencies

Python 3.11 is recommended. The repository now has one shared root dependency file, pinned to `metadrive-simulator==0.4.3`. Use one root-level environment for both the preserved baseline and the YOLO/weather prototype.

### Shared environment

```powershell
cd <repository-root>
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

On Windows CPU machines, install the official CPU PyTorch wheels after the shared requirements. This prevents GPU/DLL loading issues on machines without a compatible CUDA setup:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If activation is blocked, run this once per PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the baseline independently

Run all commands from the baseline directory because data/checkpoint paths are relative.

```powershell
cd .\baseline\bc_metadrive
```

Recommended order:

```powershell
# Diagnostics; do not start the full driving renderer
python inspect_dataset.py
python inspect_actions.py
python test_dataset.py
python test_model.py

# Manual MetaDrive collection; creates data/train/episode_*.npz
python collect_demos.py

# BC training; creates checkpoints/bc_policy.pth
python train_bc.py
```

Optional checks:

```powershell
python test_action_timing.py
python test_training_sample.py
```

`evaluate_bc.py` intentionally loads `checkpoints/bc_policy_50eps.pth`, the frozen comparison model rather than the most recently generated checkpoint.

```powershell
python evaluate_bc.py
```

`test.py`, `collect_demos.py`, and `evaluate_bc.py` use onscreen MetaDrive rendering and may fail with out-of-memory errors on a low-memory Windows laptop. Close other heavy applications and run one MetaDrive process at a time.

## Run YOLO + weather independently

Run from the repository root. The script resolves its local `yolo` and `weather` packages itself.

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather none
```

This is the current live perception demonstrator: it uses the physical
front-left/front-right pair for StereoSGBM depth and YOLO on the front, left,
right and rear logical views. Controls: `W/A/S/D` drive; `Q` quits.

```powershell
# Display weather while retaining clean stereo correspondence (default).
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather rain --level 0.5 --traffic-density 0.5

# Weather-degraded perception experiment: YOLO and StereoSGBM receive weathered frames.
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather fog --level 0.5 --perception-weather

# Finite automated smoke run; output is written to integrated_output\detection_log.csv.
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather none --max-steps 100
```

Verify the reusable stereo module before a handoff:

```powershell
Push-Location '.\YOLO+weather module'
..\venv\Scripts\python.exe .\stereo_depth\test_module.py
Pop-Location
```

### YOLO-weather unit and integration tests

The non-GUI suite checks camera calibration, SGBM configuration and depth conversion, stereo uncertainty, four-camera coordinate projection, weather augmentation, YOLO class/confidence filtering, visualization helpers, and the saved-image weather benchmark with a mocked model. It does not start MetaDrive or require `yolov8n.pt`.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s ".\YOLO+weather module\tests" -v
```

The first run downloads the official `yolov8n.pt` weights when network access is available; alternatively, provide a local weight file with `--yolo-model <path>`. Use `--device cpu` unless a compatible CUDA setup is verified.

### Saved-image weather benchmark

This benchmark runs the same YOLO model on a clean image and a synthetic fog/rain version, then writes both annotated images and `summary.csv` for a direct detection-count/confidence comparison:

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\benchmark_weather_perception.py" --image .\path\to\image.png --weather fog --level 0.5 --device cpu
```

Weather is display-only unless `--perception-weather` is specified. Even with
that flag it is synthetic image corruption, not physical weather simulation.
The first run may download `yolov8n.pt`. Use `--device cpu` unless a compatible CUDA setup is verified.

## Original BC-milestone status (historical, 2-5 Nov 2026; superseded by sections 1-11 above)

The repository now contains an end-to-end clear-condition BEV behavioural-cloning milestone: IDM-labelled MetaDrive demonstrations can be collected, used to train the CNN+MLP policy, and evaluated in closed loop. Local datasets, checkpoints, videos, and reports are deliberately Git-ignored, so their presence on one machine is not a reproducible or versioned research result. Run the module tests and repeat the documented experiments before relying on a component or reporting a performance claim. Treat the Python source plus this README as authoritative over older design notes.

| Layer | Status | Evidence / limitation |
|---|---|---|
| Preserved vector BC baseline | Present | Separate 259-D MLP experiment; not the target BEV controller. |
| Five-camera sensor rig | Present | Captures front-left/right, left, right, rear, and front SGM-depth files. |
| YOLO + stereo live demo | Present | YOLO can detect live objects; every front detection must still obtain valid SGM depth before it can support BEV. |
| 64×64 BEV/state | Partial | Assembles a 64×64 visible/free/unknown/occupied/ego grid plus the 6-D scalar state; detected objects use estimated physical footprints. Route/lane/boundary overlays are now implemented (`bev_rl/adaptive/spec_bev.py`) and the camera-side builder is `bev_state/camera_grid_builder.py`. |
| 6-D scalar state | Present | `[speed, progress, lateral deviation, heading error, curvature ahead, distance to goal]`. |
| CNN + MLP BC policy | Implemented milestone | `bev_policy/` collects IDM-labelled clear or traffic demonstrations, trains the policy, and evaluates it in closed loop. The sparse-traffic five-demo checkpoint achieved 90% success, 10% collision, and 0% off-road over 10 held-out clear sparse-traffic seeds at density 0.05. |
| Weather robustness evaluation | Preliminary tooling | The evaluator can apply paired synthetic image corruption to perception frames. It does not change vehicle dynamics or road friction, and is not evidence of physical fog/rain robustness or adaptive driving. |
| PPO and energy | Implemented, not successful | Energy model in `bev_rl/energy.py`; four PPO attempts (section 6) did not beat imitation; a validated control-effort/jerk study is not done. |

## Experiment contracts

| Experiment | Observation | Policy | Action |
|---|---|---|---|
| Preserved baseline | 259-D MetaDrive vector | MLP | `[steering, throttle/brake]` |
| Proposed BEV policy | 64 x 64 BEV + 6-D scalar state | CNN + MLP fusion | `[steering, throttle/brake]` |

Development curriculum and status:

```text
BEV behavioural cloning in clear, no-traffic MetaDrive scenarios (implemented)
  -> PPO (tried four times; none beat imitation - section 6)
  -> energy-aware accounting (implemented: bev_rl/energy.py)
  -> reduced visibility (synthetic filter; measured on the full camera pipeline - sections 5 and 7)
  -> traffic (DAgger with density 0.05; busy-traffic study at 0.15 - section 5)
```

The current closed-loop evaluator reports success, collision and off-road rates, mean speed, mean absolute steering, and a longitudinal-control-change proxy. Energy per metre, physical jerk, lane deviation, heading error, traffic robustness, and visibility-adaptive behaviour remain planned evaluation measures. Use matched MetaDrive scenarios and do not directly compare CARLA and MetaDrive metrics.

## Generated artifacts and Git

Git ignores local environments, caches, demo data (`*.npz`), non-project weights (`*.pth`, `*.onnx`), videos, local configuration (`.env`), YOLO output, and standard experiment-run directories. Project BC checkpoints (`*.pt`) are deliberately versionable so the RL stage can begin from an exact BC policy. Commit only the selected checkpoints; do not add downloaded third-party model weights without checking their licence and size.

Ignore rules do not remove a file that was already committed. Before pushing a large artifact, verify it with:

```powershell
git check-ignore -v <path>
git ls-files -- <path>
```

The first command shows the ignore rule; the second shows whether the path is already tracked.

## Controlled visibility

The evaluator accepts `--visibility-range-m` as a privileged controlled-visibility intervention that masks forward BEV cells. It is not camera-inferred physical fog range and must not be presented as adaptive weather control.

## BC-to-RL handoff baseline

The handoff package is a BC initialization, not a completed safety or weather-adaptive controller. The recommended RL starting checkpoint is `bev_policy/checkpoints/bc_sparse_traffic_5eps.pt`. It was trained from five arrival-filtered demonstrations in `bev_policy/data/sparse_traffic_5eps/` and tested in clear weather on seed-disjoint scenarios (seeds 1000--1009) with traffic density 0.05:

| Metric | Result |
|---|---:|
| Success rate | 0.90 (9/10) |
| Collision rate | 0.10 (1/10) |
| Off-road rate | 0.00 (0/10) |
| Mean speed | 8.13 m/s |

The structured report is `bev_policy/outputs/traffic_comparison/sparse_traffic_5eps/none_level_0.50_traffic_0.05_visibility_default_seed_1000_n_10.{json,csv}`. For ablations, retain `bc_clear_5eps_without_traffic.pt` as the no-traffic baseline. The RL owner should report results against both checkpoints under matched traffic, weather, and seed contracts.