# Pre-registration — written 2026-11-10 BEFORE the depth measurement, the calibrated environment or any trained model existed

Purpose: fix the hypotheses, metrics, thresholds and analysis so that a result cannot be tuned after the fact. Whatever happens is reported.

## What changed and why
The earlier weather proxy (independent random vehicle deletion, probability = measured YOLO retention) left every policy flat at density 0.05, so no adaptation claim was testable.
The user's position: the weather augmentation also degrades the **stereo depth** that places detected vehicles in the BEV. We therefore (1) MEASURE it with the teammates' own functions
against MetaDrive truth (`measure_depth_error.py`), (2) build the perturbation from the measurement (`degraded_env.py`), (3) test whether it hurts, (4) only then test adaptation.
The measurement defines the perturbation; nothing in `degraded_env.py` is tuned to make baselines fail.

## Hypotheses and decision rules (density 0.05, fresh seeds 5000+, horizon 800, identical seeds across models, exact McNemar on paired success)
* **H0 (the perturbation is real).** The measurement shows, for at least one of fog 1.0 / rain 0.6 / rain 1.0, a lower YOLO detection rate OR lower valid-depth rate OR a median absolute depth error larger than in clean frames, with non-overlapping 95% intervals for the rates.
  *If false:* weather does not degrade depth in this pipeline; the "depth degrades" premise is not supported and is reported as such.
* **H1 (it hurts a clean-trained reference).** `ov_split` (trained without any degradation, no weather input) has lower success in at least one degraded cell than in clear weather evaluated through the SAME calibrated environment, by >= 0.05 absolute (n = 100 paired seeds), exact McNemar p < 0.05.
  *If false:* at density 0.05 the measured degradation does not hurt a clean-trained policy; adaptation then cannot be demonstrated on success in sparse traffic, and the secondary metrics (below) are reported as exploratory only.
* **H2 (adaptation).** `ppo_ada` (weather input) beats `ppo_plain` (identical start, reward, data, budget, no weather input) pooled over the three degraded cells by >= 0.04 absolute success (n = 200 paired seeds per cell), exact McNemar p < 0.05 on the pooled paired outcomes, AND loses no more than 0.02 success in clear weather.
  *If false:* "adaptation shown" is NOT claimed.
* **H3 (training under degradation helps).** `deg_base` or either PPO model beats `ov_split` pooled over the three degraded cells by >= 0.04, p < 0.05.

## Secondary metrics (reported regardless, labelled exploratory if the primary tests fail)
minimum bumper gap to a vehicle ahead (privileged safety margin), mean speed, Wh/km, route completion, crash / off-road / timeout shares. Adaptation in behaviour is claimed only if the gap or speed changes with weather in the weather-conditioned model and NOT in the unconditioned model, AND success does not fall.

## Fixed protocol
* Seeds: perception measurement 7000-7039 (new); training 2000-2999; validation 3100-3139; hurt-check 5000-5099; final 5000-5199. No seed from the final block is used for any choice.
* Model selection: validation success (ties -> later iterate). Final reported model = the selected one; all checkpoints are exported and not cherry-picked after the final seeds are seen.
* All arms use the same environment code and the same per-episode perception random stream (`seed*7+1`).
* Reward v3 (`reward_v3.py`) and weights are fixed before training; no weight is changed after seeing results.
* The hand-set caution rule is NOT used anywhere.

## Declared limitations of the perturbation (cannot be removed tonight)
Only the FRONT stereo path is calibrated (range-binned detection, depth validity, relative range error). Objects outside the front field of view get only the measured detection retention (monocular side/rear cameras are unmeasured).
Errors are i.i.d. per step per object (the augmentation draws fresh noise every frame). Measurements use stationary vehicles at 5-19 m from a stationary ego, density 0.3 scenes, 40 scenes x 2 weather draws.

---
## AMENDMENT 1 — 2026-11-10 21:15, written BEFORE the corrected measurement exists
**What happened.** The first run of `measure_depth_error.py` (20:31-20:52, seeds 7000-7039) used `asm.mount_if_needed(md)` after every `md.reset(...)`. That function mounts the camera rig **once per environment object**
(`bev_state/assembler.py: _mounted_env_ids`), so scenes 2-40 had stale cameras. Symptom: stereo depth valid for only 2 of 73 clean vehicle-frames, both in scene 7000, identical in every weather condition.
The teammates' own `ground_truth_stereo_benchmark.py` calls `mount_camera_rig(env)` after every reset. **That run is INVALID and was discarded** (files kept as `*.SUSPECT_*`). A 3-scene check with the fix gave clean valid-depth 0.83 and detection 0.83.
The fitted model from the invalid run was moved aside before any trained model used it; the pipeline stage that was running on it (`hurt`, ov_split) is also discarded and must be rerun.
**What does NOT change:** hypotheses H0-H3, thresholds, seeds (7000-7039 were measured by an invalid run but no model was ever trained from it), evaluation protocol.
**Consequence for earlier work (to be reported):** `measure_paired.py` (source of `retention_table.json`, used by all earlier "weather = dropout" results) also used `asm.mount_if_needed` inside its per-scene loop,
so scenes 2-20 of that measurement likely had stale cameras too. The retention table may therefore be biased; every earlier weather result inherits that caveat until re-measured.
