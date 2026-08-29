# BEV policy: clear-condition behavioural cloning

This is the first policy milestone: a CNN processes the current 64x64 BEV and
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

The current BEV does not yet contain a route/lane/boundary overlay. Therefore,
this establishes a nominal driving pipeline only. Do not report its outcome as
weather-adaptive or obstacle-robust driving.
