**DAgger training rounds (rollout success mixes teacher and student actions, not a clean policy score)**

| policy | round | aggregate samples | train loss | rollout success |
|---|---|---|---|---|
| ov_split (ours) | 0 | 13463 | 0.0444 | 0.40 |
| ov_split (ours) | 1 | 24381 | 0.0468 | 0.50 |
| ov_split (ours) | 2 | 35023 | 0.0234 | 0.47 |
| ov_split (ours) | 3 | 47233 | 0.0005 | 0.50 |
| ov_split (ours) | 4 | 58975 | 0.0340 | 0.50 |
| ov_split (ours) | 5 | 73032 | 0.0514 | 0.40 |
| DAgger, GitHub BEV | 0 | 11388 | 0.0317 | 0.57 |
| DAgger, GitHub BEV | 1 | 24334 | 0.0456 | 0.53 |
| DAgger, GitHub BEV | 2 | 36048 | 0.0380 | 0.60 |
| DAgger, GitHub BEV | 3 | 46064 | 0.0416 | 0.53 |
| DAgger, GitHub BEV | 4 | 57497 | 0.0206 | 0.50 |
| DAgger, GitHub BEV | 5 | 66956 | 0.0456 | 0.57 |
