# Results: paired evaluation of Model A and Model C (20 seeds, pilot scale)

Seeds 5200-5219, horizon 1600 steps (160 s), traffic density 0.05, no teacher, no action noise at test time; the two arms run the same scenario with identical weather noise. Source files: `bev_rl/adaptive/results/severity_eval_20seed_snapshot/` (episodes, `MANIFEST_sha256.txt`) and `bev_rl/adaptive/results/severity_eval_20seed_analysis/` (`summary.json`, `tables.md`).

| | Clear A | Clear C | Fog 1.0 A | Fog 1.0 C | Rain 1.0 A | Rain 1.0 C |
|---|---|---|---|---|---|---|
| Success | 20/20 | **17/20** | 20/20 | 20/20 | 20/20 | 20/20 |
| Mean speed (m/s) | 7.58 | 7.13 | 7.82 | 5.57 | 7.82 | 5.50 |
| Duration (s) | 53.4 | 59.9 | 50.8 | 71.2 | 50.8 | 72.1 |
| Hard-braking share of steps | 0.035 | 0.017 | 0.027 | 0.003 | 0.027 | 0.002 |
| Wh/km, both-success pairs | 106.0 | 97.0 | 106.8 | 54.5 | 103.4 | 53.1 |
| Paired difference C - A (95 % bootstrap CI) | -9.0 (-24.8 to +5.2) | | -52.3 (-65.5 to -41.1) | | -50.3 (-61.1 to -40.4) | |
| Mean estimated caution (true) | - | 0.00 (0) | - | 0.96 (1) | - | 0.99 (1) |

## What the results support
- In severe synthetic fog and rain the adaptive policy kept 20/20 success, drove about 30 % slower, brakes hard far less often and used about 49 % less traction energy (the saving comes mostly from the lower speed).
- The image-based estimator is accurate on held-out routes (kind accuracy 0.976, fog MAE 0.045, rain MAE 0.051) and never raised caution in clear weather (false-caution rate 0.000).

## What they do not support
- **Clear weather:** C succeeded 17/20 against A's 20/20 (one collision, two stalls; estimated caution was exactly 0 throughout), a 15-percentage-point gap (exact McNemar p = 0.25, 3 A-only successes, 0 C-only; not conclusive at this scale). The energy comparison in clear weather is inconclusive (CI includes zero). **Preserving clear-weather performance is unresolved.** The cause is undetermined; hypotheses (DAgger reinforcing stopped states, clear weather under-represented, reduced training schedule) are untested.
- **Time limit:** at an 800-step limit C's slower driving counts as failure in fog and rain (A 19/20/20, C 17/15/14).
- **Generalisation:** synthetic image-space weather, simulation only, two severity extremes driven, identity caution assumption, pilot sample size.

The extension (seeds 5220-5239) is still running; it is not part of these numbers and will be reported separately and pooled.
