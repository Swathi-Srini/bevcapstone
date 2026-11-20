# Research integration and claim boundary

**Updated: 19 November 2026.** The PPO weather branch is already merged into
`main` as PR #17. This note connects it to the camera severity-adaptation work
and records which results can and cannot be compared.

## Recommended paper focus

Use the capstone brief's layered-audit question as the main research question:

> When controlled camera degradation occurs, how do detection, stereo depth,
> BEV construction, and policy actions change, and which of those changes are
> associated with closed-loop driving failures?

The strongest contribution available in this repository is a **paired,
stage-wise audit of a camera-derived BEV driving stack, combined with an
image-estimated caution intervention and closed-loop safety/energy outcomes**.
Do not claim a new detector, a new general corruption benchmark, or a proven
weather-robust RL controller. BEV corruption benchmarks and end-to-end driving
corruption studies already exist, including
[RoboBEV](https://arxiv.org/abs/2405.17426),
[RobustE2E](https://www.mdpi.com/2079-9292/13/16/3299), and the 2026
[DriveDegrade preprint](https://arxiv.org/abs/2608.29005). The distinction to
test here is the combination of explicit YOLO + stereo + hand-built BEV
intermediates, same-state stage swaps, image-derived caution, and paired
closed-loop outcomes in MetaDrive.

This is a research direction, not yet a demonstrated novelty claim. A related
work review and the completed stage-swap experiments are required before
claiming originality.

## What is integrated on `main`

| Track | Implementation and evidence | Interpretation |
|---|---|---|
| **Image-severity adaptive policy (primary paper track)** | Model A is non-adaptive. Model C uses the image-estimated severity as caution; the underlying C network was trained with oracle caution. The 20-pair paired study is in `docs/phase3/RESULTS_SUMMARY.md`. | In severe synthetic fog/rain, C completed 20/20 episodes and used less estimated traction energy while travelling more slowly. In clear weather it completed 17/20 versus A's 20/20. Clear-weather preservation is unresolved. |
| **BC-initialized PPO pilot (separate track)** | Safety-shaped PPO checkpoint and its 100-episode weather table are in `bev_policy/checkpoints/ppo_bev_adaptive_safety_v1.*` and `bev_policy/outputs/ppo_weather_generalization_20_stereo05/`. | 2,048 PPO steps; 50–65% success and 35–40% collisions across conditions. This demonstrates a functioning training/evaluation pipeline, not convergence or a reliable driver. |
| **Energy-weighted PPO candidate** | Separate checkpoint `ppo_bev_energy_adaptive_v1.zip`; metadata sets `energy_reward_weight` to 0.5. | It has not been compared to the safety checkpoint in a paired, matched evaluation. Its existence does not establish an energy improvement. |

These tracks must not be pooled into one success or energy table. The Phase 3
policy uses the frozen `CameraGridBuilder` path and an image-estimated caution
input. The PPO pilot uses `BEVDrivingEnv`, a BC action anchor, a short training
run, and a separate reward. Training steps, BEV construction, policy, scenario
contracts, and outcomes differ. The published PPO table's Wh/km values came
from the earlier 1800 kg point-force proxy and are historical until the
trace-based canonical recalculation is run; the safety-shaped PPO checkpoint
did not use an energy reward. Synthetic weather changes camera images and visibility masking,
not road friction or vehicle dynamics. The PPO weather module also supplies a
condition-derived visibility horizon to BEV construction. Therefore that study
does not demonstrate that PPO inferred weather severity from pixels.

## Required experiments before a stronger claim

1. **Complete the stage-wise audit on identical captured states.** For each
   captured scene, retain clean and weathered images, detections, depth maps,
   BEV grids, and policy inputs. Evaluate four BEVs: clean detections + clean
   depth; weathered detections + weathered depth; weathered detections + clean
   depth; and clean detections + weathered depth. Report detection misses
   separately from localization error, stereo valid fraction and distance
   error, BEV object recall/IoU, and the action change at that same state.
   Treat these swaps as offline counterfactuals, not deployable inputs.

2. **Run closed-loop matched comparisons with one observation contract.**
   Compare the fixed policy, image-caution policy, BC, PPO, and energy-weighted
   PPO only after all use the same frozen camera/BEV builder, action mapping,
   weather images, traffic density, route seeds, and episode horizon. Retrain
   any policy whose input representation changes; do not evaluate an old
   checkpoint on a newly changed BEV and call that a trained comparison.

3. **Measure camera reliance as well as image quality.** Pair graded fog/rain
   with explicit information-removal controls (e.g., a missing camera view or
   removed detections). A policy that barely changes when images are degraded
   may be robust, or may be ignoring those images; the removal control
   distinguishes these explanations. DriveDegrade makes this ambiguity
   explicit in its quality-versus-information-removal analysis.

4. **Separate safety from the slower-is-cheaper effect.** Report success,
   collision/off-road, near-miss clearance, hard braking, trip time, speed, and
   traction energy together. Add an always-cautious control and a
   speed-matched control before attributing lower Wh/km to better efficiency.
   Continue to label the current energy model as a simulator proxy, not battery
   consumption.

5. **Pre-register generalization.** Keep training, validation, and test seeds
   disjoint; include intermediate and unseen severity levels, multiple traffic
   densities, and a scenario-family holdout. Choose sample size from the
   planned estimand and uncertainty target before collecting the confirmatory
   test set. Preserve model hashes, full per-episode outcomes, and exact
   configuration for each arm.

## Current execution blocker

The Phase 3 status note planned an extension on seeds 5220–5239, but that output
directory is absent in this checkout. A local attempt to start it stopped at
the existing `verify_frozen()` check: the manifest includes source files whose
current bytes no longer match the frozen hashes. The two original policy
checkpoints were restored from byte-identical copies in `trained_models/`, but
the exact frozen `bev_state/assembler.py` version is not present in Git
history. Re-running and merging the extension without resolving this source
snapshot would weaken the paired-reproducibility claim, so no extension results
are reported here.

The PPO results from PR #17 remain available as a separate pilot. A future
unified PPO comparison needs a new explicit protocol and a retrained checkpoint
on the selected shared BEV pipeline.
