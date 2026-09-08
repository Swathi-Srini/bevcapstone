# BEV policy: clear-condition behavioural cloning

This is the policy milestone: a CNN processes the current 64x64 BEV and
an MLP processes the six documented scalar features. Their fused output is
MetaDrive's `[steering, throttle/brake]` action. It is separate from the
preserved vector baseline.

The collector uses MetaDrive `IDMPolicy` only to create clear-condition action
labels. The learnt policy receives only the BEV/state contract. This is not yet
the fog/rain result or PPO stage.

Run from the repository root after the perception-to-BEV visual test passes:

```powershell
# Smoke check: the car drives automatically; this is expensive on CPU.
python -m bev_policy.collect_demos --episodes 1 --max-steps 500 --seed 300 --render

# Actual initial clear dataset: 20 seed-disjoint episodes.
python -m bev_policy.collect_demos --episodes 20 --max-steps 500 --seed 300

# Train after at least three episodes have been collected.
python -m bev_policy.train_bc --epochs 60

# Headless metrics, then one visible held-out run.
python -m bev_policy.evaluate --episodes 10 --seed 1000
python -m bev_policy.evaluate --episodes 1 --seed 1000 --render

# Correct full-pipeline evaluation, saved for smooth playback afterwards.
python -m bev_policy.evaluate --episodes 1 --seed 1000 --record-video .\bev_policy\outputs\seed_1000.mp4
```

## Paired synthetic-weather baseline

The evaluator can now hold the learned clear checkpoint, simulator seeds,
horizon, and traffic density fixed while applying synthetic fog/rain-like image
corruption to every perception frame. The front stereo images receive the same
weather realization before SGBM, preventing artificial unmatched rain/noise
features.

Run all three conditions with `traffic-density 0.0` for the current no-traffic
experiment:

```powershell
$ckpt = .\bev_policy\checkpoints\bc_clear_5eps.pt
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --weather none
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --weather fog --weather-level 0.5
python -m bev_policy.evaluate --checkpoint $ckpt --episodes 10 --seed 1000 --max-steps 800 --weather rain --weather-level 0.5

python -m bev_policy.plot_weather_results --results `
  .\bev_policy\outputs\weather\none_level_0.50_seed_1000_n_10.json `
  .\bev_policy\outputs\weather\fog_level_0.50_seed_1000_n_10.json `
  .\bev_policy\outputs\weather\rain_level_0.50_seed_1000_n_10.json
```

Each run writes a per-episode CSV and a JSON report containing the complete
condition contract and summary. The plot shows success, off-road rate, mean
speed, and a longitudinal control-change proxy. It is not physical jerk.

This is a perception-robustness baseline, not a claim about physical fog/rain:
the simulator road friction and vehicle dynamics are unchanged. The current BEV
does not yet contain a route/lane/boundary overlay either. Therefore, do not
report these outcomes as weather-adaptive or obstacle-robust driving.

## Sparse-traffic BC handoff baseline

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