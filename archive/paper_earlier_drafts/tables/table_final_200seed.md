| arm | n | success [95% CI] | crash | off-road | timeout | speed m/s | Wh/km | route |
|---|---|---|---|---|---|---|---|---|
| safe_teacher | 200 | 0.950 [0.910, 0.973] | 0.010 | 0.040 | 0.000 | 7.84 | 101.3 | 0.968 |
| base_ovsplit | 200 | 0.925 [0.880, 0.954] | 0.060 | 0.010 | 0.005 | 7.87 | 96.2 | 0.939 |
| ctrl_nocaut | 200 | 0.895 [0.845, 0.930] | 0.105 | 0.000 | 0.000 | 7.98 | 80.4 | 0.932 |
| stock_idm | 200 | 0.885 [0.833, 0.922] | 0.080 | 0.035 | 0.000 | 7.90 | 97.9 | 0.933 |
| repo_bc_clear | 200 | 0.850 [0.794, 0.893] | 0.125 | 0.025 | 0.000 | 8.09 | 85.9 | 0.918 |
| repo_bc_sparse | 200 | 0.745 [0.680, 0.800] | 0.120 | 0.135 | 0.000 | 8.07 | 90.7 | 0.883 |

Paired (same seeds): diff = success(A) - success(B); b/c = discordant seeds (A only / B only); exact McNemar p; 95% paired-bootstrap CI.

| A | B | diff | A-only | B-only | p | boot CI |
|---|---|---|---|---|---|---|
| safe_teacher | base_ovsplit | +0.025 | 14 | 9 | 0.4049 | [-0.025, +0.075] |
| safe_teacher | ctrl_nocaut | +0.055 | 20 | 9 | 0.0614 | [+0.005, +0.105] |
| safe_teacher | stock_idm | +0.065 | 15 | 2 | 0.0023 | [+0.030, +0.105] |
| safe_teacher | repo_bc_clear | +0.100 | 24 | 4 | 0.0002 | [+0.050, +0.150] |
| safe_teacher | repo_bc_sparse | +0.205 | 44 | 3 | 0.0000 | [+0.145, +0.265] |
| base_ovsplit | ctrl_nocaut | +0.030 | 10 | 4 | 0.1796 | [-0.005, +0.065] |
| base_ovsplit | stock_idm | +0.040 | 12 | 4 | 0.0768 | [+0.000, +0.080] |
| base_ovsplit | repo_bc_clear | +0.075 | 17 | 2 | 0.0007 | [+0.035, +0.120] |
| base_ovsplit | repo_bc_sparse | +0.180 | 38 | 2 | 0.0000 | [+0.125, +0.240] |
| ctrl_nocaut | stock_idm | +0.010 | 9 | 7 | 0.8036 | [-0.030, +0.050] |
| ctrl_nocaut | repo_bc_clear | +0.045 | 11 | 2 | 0.0225 | [+0.010, +0.080] |
| ctrl_nocaut | repo_bc_sparse | +0.150 | 30 | 0 | 0.0000 | [+0.100, +0.200] |
| stock_idm | repo_bc_clear | +0.035 | 10 | 3 | 0.0923 | [+0.000, +0.070] |
| stock_idm | repo_bc_sparse | +0.140 | 30 | 2 | 0.0000 | [+0.090, +0.195] |
| repo_bc_clear | repo_bc_sparse | +0.105 | 25 | 4 | 0.0001 | [+0.055, +0.155] |