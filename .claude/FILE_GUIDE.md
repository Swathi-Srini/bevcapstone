# File and folder guide (plain-English names for the short run names)
Last updated 2026-11-04. Short names are kept on disk because scripts hard-code them
(renaming mid-sweep would break running jobs). New files/folders get descriptive names.

## LATEST MODEL TO USE
D:\bev_Fresh\bevcapstone\bev_rl\adaptive\runs\ov_split\best.pt
  = "Imitation-learned (DAgger) driving policy: overlay BEV (route + road boundary) + 3 scalars
     (speed, route progress, distance to goal)". Best validation iteration (iter 5), saved 2026-11-04 03:38.
  Reserved-seed result (n=10): success 0.80 at traffic density 0.05, 0.60 at 0.20. Starting point for all PPO / caution work.
  Metrics: runs/ov_split/best.json ; reserved-seed eval: results/final_ov_split_best.json

## Run folders (bev_rl/adaptive/runs/<short name>/ : best.pt = chosen, iter_N.pt = per round, log.json)
| Short name        | Meaning |
|-------------------|---------|
| base_a            | DAgger baseline: GitHub BEV + all 6 scalars |
| arch_bev_only     | GitHub BEV only, no scalars (0 success; failed) |
| arch_scalar_only  | Scalars only, BEV zeroed |
| ov_both           | Overlay BEV + all 6 scalars |
| ov_bev_only       | Overlay BEV, no scalars |
| ov_split          | Overlay BEV + 3 scalars  <-- BEST, the base for everything newer |
| caut_a            | Caution-conditioned DAgger (caution input + caution-aware teacher). Stopped at iter 1 of 5 by shutdown |
| ctrl_nocaut       | Control for caut_a (caution forced to 0, stock teacher). Stopped at iter 0 by shutdown |
| rl_full           | PPO from ov_split: energy + safe-speed + heading + crash-cost rewards (sweep, running) |
| rl_plain / rl_energy / rl_safe / rl_head | PPO reward ablations (queued after rl_full) |
| pilot_cond, probe_C, C2, C3, smoke | Older experiments / smoke tests, not used for results |

## Other
- results/*.json|csv : measured numbers (final_*_best.json = reserved-seed evals; retention_table / perception_* = detector degradation)
- results/caution/base_ovsplit.json : ov_split evaluated with the caution eval harness (reference arm)
- logs/ : training/eval logs, one per run
- ../../paper/ : outline, figures, references (paper/PAPER_OUTLINE.md, paper/figures/)
- .claude/PROJECT_STATE.md : full decision + results log
