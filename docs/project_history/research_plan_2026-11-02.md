# Research Plan

**Working title:** Weather-conditioned reinforcement learning for autonomous driving with camera-derived BEV perception under fog and rain (MetaDrive).

## Research questions
- **RQ1 (perception).** How much does a camera-only, geometry-grounded BEV (road, lane line, boundary, obstacle, explicit unknown) degrade under depth-dependent fog and seeded rain, and which BEV components degrade first?
- **RQ2 (policy).** Does explicit weather conditioning (type + intensity) improve a PPO policy's safety and route completion under fog/rain versus (a) the same architecture without the weather input and (b) a clear-weather-trained policy?
- **RQ3 (generalisation).** Do conditioned policies generalise to unseen intensities, road layouts and seeds? Does mixed-weather training matter more than conditioning?
- **RQ4 (BEV value/cost).** What does the camera-derived BEV cost (latency, memory), and how do policies on it compare with an oracle (simulator-lane) BEV?

## Hypotheses (to be tested, not assumed)
- H1: BEV known-fraction and boundary/lane quality fall with fog/rain intensity (pilot: consistent at fog 1.0 and rain 1.0; intensity 0.45 nearly unchanged).
- H2: Weather-conditioned > unconditioned under degraded perception. May be null; report either way.
- H3: Mixed-weather training > clear-only training on fog/rain evaluation.

## Candidate contributions (novelty NOT claimed until LITERATURE_REVIEW supports it)
1. Reproducible camera-derived BEV pipeline for MetaDrive with explicit unknown regions and a ground-truth validation protocol.
2. Depth-dependent fog / seeded rain perturbation model with documented limits and measured effect on BEV quality.
3. Controlled study of explicit weather conditioning vs ablations for PPO on that BEV, multi-seed statistics.

## Milestones
| # | Milestone | Status |
|---|---|---|
| M0 | BEV defect audit, BEV v2, 1-seed pilot validation | done |
| M1 | Perception hardening: route-offset check, obstacle projection, multi-seed validation, lane-line classifier | in progress |
| M2 | Env integration (BEV v2 obs, weather branch), cost benchmark | |
| M3 | BC on BEV v2 (demonstration reuse or recollection) | |
| M4 | Reward audit on new env; clear-weather PPO | |
| M5 | Mixed-weather PPO; conditioned vs unconditioned; oracle-BEV baseline | |
| M6 | Evaluation: unseen intensity/layout/seed; statistics; failure analysis | |
| M7 | Paper draft, figures, tables | |

## Protocol (summary)
Seed splits are disjoint: calibration 3000–3003 (classifier), evaluation 1000–1009 (reserved), training seeds to be chosen from >= 2000 outside calibration. >= 3 training seeds per method. Metrics: success, route completion, collision, off-road, mean speed, travel time, lateral deviation, heading stability, jerk, estimated energy; mean with bootstrap 95% CI. The actor sees only the camera-derived BEV, the ego-state/navigation scalars, and the explicit weather input.
