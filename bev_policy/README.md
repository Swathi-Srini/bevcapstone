# BEV policy: YOLO-to-BEV behavioural cloning and PPO

This is the policy milestone: a CNN processes the current 64x64 BEV and
an MLP processes the six documented scalar features. Their fused output is
MetaDrive's `[steering, throttle/brake]` action. It is separate from the
preserved vector baseline. The observation path runs YOLO on four camera views,
uses front stereo depth plus side/rear ground-plane projection, then paints
detected objects and weather-limited visibility into the policy BEV.

The collector uses MetaDrive `IDMPolicy` only to create clear-condition action
labels; optional weather randomization augments the camera inputs, while the
expert action remains the MetaDrive IDM action. The PPO trainer warm-starts its
BEV/scalar encoders and action-mean layers from a BC checkpoint.

The PPO actor follows the architecture diagram: the 64x64 BEV CNN produces
128 features, the six-scalar MLP produces 32, and their 160-feature fusion
feeds a 128-unit actor and two tanh-bounded controls. PPO uses gSDE for
continuous-action exploration; a separate 128-unit value branch estimates
the critic. The environment returns MetaDrive reward and the next observation
for PPO updates.

The checked-in handoff BC checkpoints predate the current-lane center/edge BEV
overlay. They are useful for smoke tests and PPO initialization, but collect
and train a fresh BC checkpoint with the current code before reporting
comparable policy results.

Run from the repository root after the perception-to-BEV visual test passes:

```powershell
# Smoke check: the car drives automatically; this is expensive on CPU.
python -m bev_policy.collect_demos --episodes 1 --max-steps 500 --seed 300 --render

# Fresh clear-weather, sparse-traffic BC pilot: five seed-disjoint episodes.
python -m bev_policy.collect_demos --episodes 5 --max-steps 500 --seed 300 `
  --traffic-density 0.05 --output-dir .\bev_policy\data\bc_clear_sparse_5eps

# Train after at least three episodes have been collected.
python -m bev_policy.train_bc --data-dir .\bev_policy\data\bc_clear_sparse_5eps `
  --checkpoint .\bev_policy\checkpoints\bc_clear_sparse_5eps.pt --epochs 60 --seed 2026

# Paired held-out clear-condition pilot evaluation.
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\bc_clear_sparse_5eps.pt `
  --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05

# Interactive run with the combined driving/perception window.
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\bc_clear_sparse_5eps.pt `
  --episodes 1 --max-steps 200 --seed 1000 --traffic-density 0.35 `
  --weather clear --weather-level 0 --inference-size 640 `
  --confidence-threshold 0.25 --render

# Correct full-pipeline evaluation, saved for smooth playback afterwards.
python -m bev_policy.evaluate --episodes 1 --seed 1000 --record-video .\bev_policy\outputs\seed_1000.mp4
```

BC checkpoints record the seed, hyperparameters, and episode-level
train/validation split. Reusing the same dataset and seed reproduces model
initialization and shuffled batch order.
The five-episode set is a pipeline pilot, not enough evidence for a paper
performance claim; collect a larger dataset and repeat held-out evaluations.

Pilot audit: seeds 300--304 produced 1,919 aligned samples and all five IDM
episodes reached the destination, but none contained a braking label
(`action[1] < -0.05`). The seeded 60-epoch run reached a best validation loss
of 0.00930 on the single held-out episode. The policy then collided on held-out
seed 1000 at step 323. This verifies collection, training, checkpoint loading,
and closed-loop execution; it is not evidence of safe or robust driving.

### Braking-label follow-up

A braking-qualified follow-up dataset is in
`bev_policy/data/bc_clear_braking_traffic_5eps/`. The current three accepted
episodes (seeds 600, 609, and 612; traffic density 0.1) contain 1,385 aligned
samples and 23 braking labels; all three IDM demonstrations reached the
destination. Training used two episodes and held one out, with 15 train and
eight validation braking labels. The seeded, brake-weighted run stopped at
epoch 26 with best validation loss 0.04491. The quality gate
`--min-brake-samples 5` was enabled.

This fixes the pilot dataset's missing-brake-label problem, but **does not make
the BC policy ready for safety or research claims**. On held-out clear-weather
seeds 1000--1002 at traffic density 0.1, the checkpoint went off-road on all
three episodes (steps 94, 253, and 84 respectively). The three-episode dataset
is only an integration pilot; gather substantially more seed-diverse
demonstrations and improve closed-loop recovery before treating this policy as
a reliable baseline.

#### Expanded follow-up (2026-11-17)

The same dataset folder now contains seven accepted IDM episodes (seeds
600, 609, 612, 621, 627, 704, 707), collected at traffic densities 0.1 and
0.2. They contain 2,964 aligned samples and 49 braking labels; every saved
episode reached its destination and had projected-object frames. Rejected
crash, timeout, and insufficient-braking episodes were not included. The
episode-level split is six training episodes and one validation episode, so
validation evidence remains limited.

The brake loss and brake-data quality gate use the same threshold
(`action[1] < -0.05`). Fine-tuning from an existing checkpoint now preserves
that checkpoint's scalar normalization; previously weights could be loaded
while their inputs were silently renormalized. Seven BC unit tests cover the
braking threshold, data gate, deterministic training, and warm-start
normalization.

Three newly trained candidates (scratch, warm start, and normalization-
preserving warm start) each had 0/5 successes on the same held-out clear
seeds 1000--1004 at traffic density 0.1; all five episodes went off-road.
A conservative fine-tune also had 0/5 success, all off-road. The earlier
sparse-traffic checkpoint scored 2/5 success, 2/5 collisions, 0 off-road, and
1/5 max-step on those same seeds. IDM reached the destination on 3/5 seeds
(the other two timed out). Therefore **the expanded-data BC candidates are
not ready to use**; use none as a safe-driving claim or paper result.

Held-out reports are saved under
`bev_policy/outputs/bc_braking_mixed_traffic_warmstart_test/`,
`bev_policy/outputs/bc_braking_mixed_traffic_scratch_test/`,
`bev_policy/outputs/bc_braking_mixed_traffic_preserve_norm_test/`, and
`bev_policy/outputs/bc_braking_conservative_finetune_test/`. The older
checkpoint comparison is under
`bev_policy/outputs/bc_sparse_traffic_old_checkpoint_test/`.

The braking-weighted, normalization-preserving checkpoint can be reproduced:

```powershell
python -m bev_policy.train_bc --data-dir .\bev_policy\data\bc_clear_braking_traffic_5eps `
  --checkpoint .\bev_policy\checkpoints\bc_braking_mixed_traffic_preserve_norm.pt `
  --epochs 60 --seed 2026 --min-brake-samples 5 --brake-weight 5 `
  --init-checkpoint .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt
python -m bev_policy.evaluate `
  --checkpoint .\bev_policy\checkpoints\bc_braking_mixed_traffic_preserve_norm.pt `
  --episodes 5 --seed 1000 --max-steps 500 --traffic-density 0.1 `
  --weather clear --weather-level 0 --inference-size 640 `
  --output-dir .\bev_policy\outputs\bc_braking_mixed_traffic_preserve_norm_test
```

With `--render`, evaluation opens one combined window: the MetaDrive chase
view, the policy BEV, YOLO boxes on the front-left/left/right/rear views, the
front-right stereo partner, and the front StereoSGBM depth map. The images
shown are the same weather-conditioned images used by YOLO and the policy.
Press `Q` in the combined window to stop. `--no-perception-view` disables the
combined window and uses MetaDrive's normal render window instead.
For visible traffic detections, start with `--traffic-density 0.35`,
`--inference-size 640`, and `--confidence-threshold 0.25`. Sparse traffic,
small inference images, or heavy weather can legitimately produce zero YOLO
boxes in a frame; the dashboard reports per-camera counts and stereo-depth
coverage rather than drawing fabricated detections.

Compare clear and weather-degraded perception using the same checkpoint and
seed:

```powershell
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt `
  --episodes 1 --max-steps 200 --seed 1000 --traffic-density 0.35 `
  --weather clear --weather-level 0 --inference-size 640 `
  --confidence-threshold 0.25 --render
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt `
  --episodes 1 --max-steps 200 --seed 1000 --traffic-density 0.35 `
  --weather fog --weather-level 0.8 --inference-size 640 `
  --confidence-threshold 0.25 --render
```

## Configurable weather experiments

Weather intensity is configurable in `[0,1]`. Fog uses depth-dependent
Beer-Lambert atmospheric attenuation and atmospheric light. Rain adds
distance-dependent veiling, exposure streaks, wet-road darkening, and
camera-lens droplets. The front stereo pair receives matched stochastic
weather, and each realization is seeded so repeated evaluations use the same
weather for the same episode seed. A conservative horizon from the weather
model marks BEV cells as unknown; `--visibility-range-m` overrides it.

This is a physically motivated camera-image approximation, not MetaDrive
ray-traced weather, measured precipitation, or a calibrated visibility sensor.
Vehicle dynamics and road friction are intentionally unchanged. Results must
be described as weather-conditioned perception experiments, not proof of
physical weather robustness.

To collect IDM labels from randomized weather for weather-augmented BC, then
train a separate checkpoint:

```powershell
python -m bev_policy.collect_demos --episodes 20 --seed 500 --random-weather `
  --weather-level 0.8
python -m bev_policy.train_bc --data-dir .\bev_policy\data\weather_randomized `
  --checkpoint .\bev_policy\checkpoints\bc_weather.pt --epochs 60
```

The collector stores the sampled weather and level in each episode's metadata.
Its default randomized dataset folder is `bev_policy/data/weather_randomized`;
set `--output-dir` to isolate additional runs.

```powershell
$ckpt = .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 --weather clear --weather-level 0.0
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 --weather fog --weather-level 0.5
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 --weather rain --weather-level 0.5

python -m bev_policy.plot_weather_results --results `
  .\bev_policy\outputs\weather\clear_level_0.00_traffic_0.05_visibility_default_seed_1000_n_10.json `
  .\bev_policy\outputs\weather\fog_level_0.50_traffic_0.05_visibility_default_seed_1000_n_10.json `
  .\bev_policy\outputs\weather\rain_level_0.50_traffic_0.05_visibility_default_seed_1000_n_10.json
```

Each run writes a per-episode CSV and a JSON report containing the complete
condition contract, visibility horizon, and summary. The plot shows success,
off-road rate, mean speed, and a longitudinal control-change proxy. It is not
physical jerk. For randomized testing, use `--weather mixed --random-weather
--weather-level 0.8`; the condition and intensity are sampled per episode.

## BC-initialized PPO adaptation experiment

The PPO stage samples clear/fog/rain conditions and intensities in
`[0, --weather-level]` by default, learns from MetaDrive rewards, and is
initialized from the BEV BC checkpoint. It is an experiment scaffold: a small
smoke run verifies execution, but meaningful adaptive-driving claims require
long training, paired held-out seeds, and the reported safety metrics.
Each episode now receives a reproducible new MetaDrive seed instead of
restarting the same scenario. PPO fine-tuning uses a frozen BC policy as an
action anchor: an auxiliary action-matching loss discourages updates from
drifting far from BC, alongside a lower learning rate, fewer PPO epochs, and
a KL early-stop threshold. Start with a clear-weather pilot and compare against
BC on identical held-out seeds before evaluating weather or extending training.

```powershell
# Bounded integration smoke test; the full camera/YOLO loop is compute-heavy.
python -m bev_policy.train_ppo --timesteps 256 --n-steps 64 --batch-size 32 `
  --max-steps 100 --inference-size 320 --weather-level 0.8 `
  --output .\bev_policy\checkpoints\ppo_smoke

# Research run (increase timesteps and evaluate multiple held-out seeds).
python -m bev_policy.train_ppo --timesteps 100000 --weather-level 0.8 `
  --output .\bev_policy\checkpoints\ppo_adaptive

# Same held-out seeds/condition for the BC and PPO policies.
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt `
  --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 --weather fog --weather-level 0.5
python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\ppo_adaptive.zip `
  --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 --weather fog --weather-level 0.5
```

`evaluate` detects `.zip` checkpoints as PPO and loads the accompanying JSON
normalization metadata. Repeat the paired comparison for clear and rain, then
vary `--weather-level` (for example 0.25, 0.5, 0.8). Report raw per-episode
outcomes; a brief smoke run is not evidence that PPO improved adaptation.

## Current project stage and saved pilot results

The end-to-end BEV behavioural-cloning path is implemented. Weather-conditioned
perception and BEV visibility masking are also implemented as synthetic camera
weather experiments. PPO training runs, but the saved training runs are only
512 steps and do not demonstrate improved driving or adaptive caution. Do not
describe the project as having a trained weather-adaptive RL driver yet.

The latest locally saved matched clear evaluation on seeds 1000--1009 at
traffic density 0.05 records 8/10 success, 1/10 collision, and 1/10 off-road
for BC; the 512-step BC-anchored PPO checkpoint records the same outcomes on
the same seeds. Reports are under
`bev_policy/outputs/comparison/bc_clear_anchor_v1/` and
`bev_policy/outputs/comparison/ppo_clear_anchor_v1/`. These reports date from
12 November 2026 and predate the current lane-center/edge BEV changes, so they
are historical smoke evidence and need to be reproduced with the current code.
A separate randomized-weather PPO checkpoint had 0/5 successes in its saved
clear pilot. None of these results show a PPO gain.

The BC checkpoint also has a historical 10-seed fog-level-0.5 report with
9/10 success, 1/10 collision, and 0/10 off-road, generated 13 November 2026.
It predates the current BEV changes and does not prove adaptation: there is no
paired PPO fog/rain evaluation or controlled speed/braking analysis showing
that lower visibility caused safer behaviour.

Current-code full-pipeline BEV PPO replays use the saved checkpoint
`ppo_bev_bc_anchor_v1.zip`, initialized from `bc_sparse_traffic_5eps.pt` and
fine-tuned for 512 PPO steps. On matched seed 1000 at traffic density 0.05,
clear, fog level 0.5, and rain level 0.5 each reached the destination in 369
steps with no collision or off-road event. Mean speeds were 8.33, 8.30, and
8.32 m/s respectively, so the runs do not show a meaningful cautious response
to weather. They are single-seed demonstrations, not a weather-robustness
claim. Full dashboard videos and reports are under
`bev_policy/outputs/weather_pipeline_demo/`; the dashboard includes simulator
view, BEV, four YOLO views, front stereo partner, stereo depth, action, and
weather label. Weather is synthetic camera-image corruption; road friction and
vehicle dynamics are unchanged. The weather runs used `--stereo-scale 0.5` to
reduce CPU cost while preserving a full-camera-sized depth output.

To record the full perception dashboard, pass `--record-dashboard <path.mp4>`
to `bev_policy.evaluate`. Stereo matching remains full resolution by default;
use `--stereo-scale 0.5` for the faster half-resolution matcher used in these
CPU replays.

To complete the stated adaptive-driving objective, the remaining work is:

1. Train for substantially longer than 512 steps with randomized weather and
   retain the BC anchor until PPO shows stable improvement.
2. Evaluate BC and PPO on identical held-out seeds in clear, fog, and rain at
   multiple severity levels and light traffic; preserve per-episode reports.
3. Demonstrate lower speed or earlier braking in reduced visibility while
   maintaining destination success and reducing collision/off-road outcomes.
4. Repeat promising results over a larger seed set before calling the policy
   robust. Camera/YOLO evaluation remains slow on CPU; the one-seed replay was
   practical at inference size 320, while the three-episode size-640 pilot did
   not finish within an interactive run.

#### Higher-traffic PPO pilot (2026-11-17)

A new candidate was trained for 2,048 PPO steps from `bc_sparse_traffic_5eps.pt`
with clear/fog/rain randomized by episode (maximum intensity 0.8), traffic
density 0.10, inference size 320, and YOLO confidence 0.25. Its held-out clear
evaluation on seed 1000 collided at step 317. The paired BC checkpoint collided
on the same scenario at step 319. This one-seed test shows no RL improvement;
do not use the candidate as evidence of generalization or safe driving. The
model and metadata are `checkpoints/ppo_bev_research_weather_traffic_v1.zip`
and `.json`; paired reports are under `outputs/research_train_eval/`.

The evaluator saves per-camera YOLO box counts, frames with detections and
projected BEV objects, mean stereo valid fraction, lateral/heading deviation,
and canonical Wh/km from `bev_rl/energy.py`. This is estimated mechanical
traction energy, not measured battery energy; regeneration and road grade are
omitted. The older PPO weather tables used a separate 1800 kg proxy and remain
historical until their trace-capture recalculation is run.

The BC-vs-safety PPO comparison was repeated using corrected weathered-image
stereo depth. Across the matched rain/fog/clear seeds 1000--1002 at traffic
density 0.10, safety PPO recorded 2/3 successes, 1/3 collision, and no
off-road events; BC recorded 1/3 successes and 2/3 collisions, also no
off-road events. Both collided in fog, and fog mean speed was similar
(7.86 vs 7.97 m/s). A separate energy-reward PPO also scored 2/3, but its mean
estimated Wh/km (154.0) was slightly above safety PPO (151.7) across these
three episodes. These are small, noisy pilots, not proof of generalization or
energy improvement. Their old Wh/km values used the 1800 kg point-force
implementation and must not be compared with canonical results. Corrected reports are under
`outputs/adaptive_safety_generalization_corrected/`; the previous
`outputs/adaptive_safety_generalization/` weather reports used clean stereo
depth and are superseded.

The energy-reward candidate is saved as
`checkpoints/ppo_bev_energy_adaptive_v1.zip`. Reproduce its training with:
The existing checkpoint was trained with the prior 1800 kg reward proxy;
the training source now uses the canonical `bev_rl/energy.py` implementation.

```powershell
python -m bev_policy.train_ppo --timesteps 2048 --n-steps 128 --batch-size 64 `
  --n-epochs 4 --learning-rate 0.00003 --target-kl 0.01 --bc-loss-coef 0.5 `
  --adaptive-speed-penalty 0.05 --collision-penalty 30 `
  --energy-reward-weight 0.5 --max-steps 400 --seed 2028 `
  --traffic-density 0.10 --random-weather --weather-level 0.8 `
  --inference-size 320 --confidence-threshold 0.25 `
  --output .\bev_policy\checkpoints\ppo_bev_energy_adaptive_v1
```

Reproduce the training and paired pilot comparison:

```powershell
python -m bev_policy.train_ppo --timesteps 2048 --n-steps 128 --batch-size 64 `
  --max-steps 300 --seed 2026 --traffic-density 0.10 --random-weather `
  --weather-level 0.8 --inference-size 320 --confidence-threshold 0.25 `
  --output .\bev_policy\checkpoints\ppo_bev_research_weather_traffic_v1

python -m bev_policy.evaluate --checkpoint .\bev_policy\checkpoints\ppo_bev_research_weather_traffic_v1.zip `
  --episodes 1 --max-steps 500 --seed 1000 --traffic-density 0.10 `
  --weather clear --weather-level 0 --inference-size 320 `
  --confidence-threshold 0.25 --stereo-scale 0.5
```

Earlier sparse-traffic BC handoff baseline
------------------------------------------

Five accepted sparse-traffic demonstrations are available in `bev_policy/data/sparse_traffic_5eps/`. Their checkpoint is `bev_policy/checkpoints/bc_sparse_traffic_5eps.pt`.

Reproduce the held-out evaluation (10 clear scenarios, seeds 1000--1009, traffic density 0.05):

```powershell
.\venv\Scripts\python.exe -m bev_policy.evaluate `
  --checkpoint .\bev_policy\checkpoints\bc_sparse_traffic_5eps.pt `
  --episodes 10 --seed 1000 --max-steps 800 --traffic-density 0.05 `
  --output-dir .\bev_policy\outputs\traffic_comparison\sparse_traffic_5eps
```

Observed result: 9/10 success, 1/10 collision, 0/10 off-road, and mean speed 8.13 m/s. This is suitable as an RL initialization and comparison baseline, not a claim of robust traffic safety. For the no-traffic ablation, use `bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt`.

### Commit selected handoff checkpoints

`.pt` checkpoints are no longer Git-ignored. Before committing, verify their size and add only the intended models:

```powershell
git add bev_policy/checkpoints/bc_clear_5eps_without_traffic.pt
git add bev_policy/checkpoints/bc_sparse_traffic_5eps.pt
git status
```

Demonstrations (`*.npz`) and generated reports remain ignored; share them as a separate archive or dataset release with the RL owner.

## 100-episode held-out weather study

The reproducible evaluation table and episode-level results are in
`outputs/ppo_weather_generalization_20_stereo05/`. This study ran 20 held-out
seeds for each of clear, light/heavy synthetic fog, and light/heavy synthetic
rain using the saved `checkpoints/ppo_bev_adaptive_safety_v1.zip` checkpoint.
The checkpoint was trained for 2,048 PPO timesteps from the BEV behavior-cloning
anchor, so these results are a finite pilot and do not demonstrate PPO
convergence. Success ranged from 50% to 65% (10–13 of 20) across conditions.
The report also gives raw YOLO box-instance counts and per-frame rates; these
are predictions, not unique tracked vehicles or detection accuracy. Synthetic
weather changes camera appearance and supplies the matching visibility horizon;
vehicle friction/dynamics were not changed. Energy is a modeled tractive-work
proxy, not measured battery consumption. See the table for full limitations
and confidence intervals.

The study runner uses fixed settings recorded in its source and reports. To\nreproduce the held-out study with the same saved checkpoint and output folder:\n\n```powershell\npython -m bev_policy.run_weather_generalization_study\n```\n
