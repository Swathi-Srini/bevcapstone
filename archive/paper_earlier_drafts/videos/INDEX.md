# Videos of every model (10 fps = real time; left: top-down world, right: the BEV the policy receives)

Environment: sparse traffic (density 0.05), horizon 800, detector dropout for fog/rain from the measured retention table (NO weather physics). Identical scenario seed across models in each group.

| file | model | seed | weather | outcome | steps | mean speed m/s |
|---|---|---|---|---|---|---|
| arch_bev_only__seed1012__clear.mp4 | arch_bev_only | 1012 | clear:0.0 | off_road | 89 | 1.97 |
| arch_scalar_only__seed1012__clear.mp4 | arch_scalar_only | 1012 | clear:0.0 | success | 464 | 7.24 |
| base_a__seed1012__clear.mp4 | base_a | 1012 | clear:0.0 | crash | 374 | 5.99 |
| caut_c0__seed1012__clear.mp4 | caut_c0 | 1012 | clear:0.0 | success | 465 | 7.22 |
| ctrl_nocaut__seed1012__clear.mp4 | ctrl_nocaut | 1012 | clear:0.0 | success | 466 | 7.23 |
| ov_bev_only__seed1012__clear.mp4 | ov_bev_only | 1012 | clear:0.0 | crash | 264 | 7.57 |
| ov_both__seed1012__clear.mp4 | ov_both | 1012 | clear:0.0 | success | 465 | 7.22 |
| ov_split__seed1012__clear.mp4 | ov_split | 1012 | clear:0.0 | success | 479 | 7.16 |
| repo_bc_clear__seed1012__clear.mp4 | repo_bc_clear | 1012 | clear:0.0 | success | 462 | 7.31 |
| repo_bc_sparse__seed1012__clear.mp4 | repo_bc_sparse | 1012 | clear:0.0 | success | 463 | 7.22 |
| safe_student__seed1012__clear.mp4 | safe_student | 1012 | clear:0.0 | success | 465 | 7.22 |
| safe_teacher__seed1012__clear.mp4 | safe_teacher | 1012 | clear:0.0 | success | 465 | 7.22 |
| stock_idm__seed1012__clear.mp4 | stock_idm | 1012 | clear:0.0 | success | 465 | 7.22 |
| arch_bev_only__seed1020__clear.mp4 | arch_bev_only | 1020 | clear:0.0 | off_road | 90 | 2.05 |
| arch_scalar_only__seed1020__clear.mp4 | arch_scalar_only | 1020 | clear:0.0 | success | 300 | 6.83 |
| base_a__seed1020__clear.mp4 | base_a | 1020 | clear:0.0 | success | 299 | 6.82 |
| caut_c0__seed1020__clear.mp4 | caut_c0 | 1020 | clear:0.0 | success | 298 | 6.81 |
| ctrl_nocaut__seed1020__clear.mp4 | ctrl_nocaut | 1020 | clear:0.0 | success | 300 | 6.81 |
| ov_bev_only__seed1020__clear.mp4 | ov_bev_only | 1020 | clear:0.0 | off_road | 157 | 5.58 |
| ov_both__seed1020__clear.mp4 | ov_both | 1020 | clear:0.0 | success | 336 | 6.54 |
| ov_split__seed1020__clear.mp4 | ov_split | 1020 | clear:0.0 | success | 313 | 6.21 |
| repo_bc_clear__seed1020__clear.mp4 | repo_bc_clear | 1020 | clear:0.0 | success | 302 | 6.80 |
| repo_bc_sparse__seed1020__clear.mp4 | repo_bc_sparse | 1020 | clear:0.0 | off_road | 151 | 5.90 |
| safe_student__seed1020__clear.mp4 | safe_student | 1020 | clear:0.0 | success | 302 | 6.82 |
| safe_teacher__seed1020__clear.mp4 | safe_teacher | 1020 | clear:0.0 | success | 301 | 6.82 |
| stock_idm__seed1020__clear.mp4 | stock_idm | 1020 | clear:0.0 | success | 301 | 6.82 |
| base_a__seed1039__clear.mp4 | base_a | 1039 | clear:0.0 | crash | 318 | 6.87 |
| caut_c0__seed1039__clear.mp4 | caut_c0 | 1039 | clear:0.0 | crash | 318 | 6.88 |
| ctrl_nocaut__seed1039__clear.mp4 | ctrl_nocaut | 1039 | clear:0.0 | success | 378 | 7.05 |
| ov_split__seed1039__clear.mp4 | ov_split | 1039 | clear:0.0 | success | 377 | 7.07 |
| repo_bc_clear__seed1039__clear.mp4 | repo_bc_clear | 1039 | clear:0.0 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__clear.mp4 | repo_bc_sparse | 1039 | clear:0.0 | off_road | 241 | 6.66 |
| safe_student__seed1039__clear.mp4 | safe_student | 1039 | clear:0.0 | crash | 319 | 6.88 |
| safe_teacher__seed1039__clear.mp4 | safe_teacher | 1039 | clear:0.0 | success | 415 | 6.98 |
| stock_idm__seed1039__clear.mp4 | stock_idm | 1039 | clear:0.0 | crash | 319 | 6.90 |
| base_a__seed1040__clear.mp4 | base_a | 1040 | clear:0.0 | crash | 82 | 4.68 |
| caut_c0__seed1040__clear.mp4 | caut_c0 | 1040 | clear:0.0 | crash | 82 | 4.59 |
| ctrl_nocaut__seed1040__clear.mp4 | ctrl_nocaut | 1040 | clear:0.0 | success | 386 | 7.08 |
| ov_split__seed1040__clear.mp4 | ov_split | 1040 | clear:0.0 | crash | 80 | 4.62 |
| repo_bc_clear__seed1040__clear.mp4 | repo_bc_clear | 1040 | clear:0.0 | crash | 81 | 4.71 |
| repo_bc_sparse__seed1040__clear.mp4 | repo_bc_sparse | 1040 | clear:0.0 | success | 384 | 7.12 |
| safe_student__seed1040__clear.mp4 | safe_student | 1040 | clear:0.0 | crash | 82 | 4.66 |
| safe_teacher__seed1040__clear.mp4 | safe_teacher | 1040 | clear:0.0 | success | 445 | 6.21 |
| stock_idm__seed1040__clear.mp4 | stock_idm | 1040 | clear:0.0 | crash | 81 | 4.69 |
| base_a__seed1020__fog1.0.mp4 | base_a | 1020 | fog:1.0 | success | 299 | 6.82 |
| caut_auto__seed1020__fog1.0.mp4 | caut_auto | 1020 | fog:1.0 | success | 387 | 5.52 |
| caut_c0__seed1020__fog1.0.mp4 | caut_c0 | 1020 | fog:1.0 | success | 298 | 6.80 |
| ctrl_nocaut__seed1020__fog1.0.mp4 | ctrl_nocaut | 1020 | fog:1.0 | success | 300 | 6.82 |
| ov_split__seed1020__fog1.0.mp4 | ov_split | 1020 | fog:1.0 | success | 313 | 6.21 |
| repo_bc_sparse__seed1020__fog1.0.mp4 | repo_bc_sparse | 1020 | fog:1.0 | off_road | 151 | 5.90 |
| base_a__seed1039__fog1.0.mp4 | base_a | 1039 | fog:1.0 | crash | 318 | 6.87 |
| caut_auto__seed1039__fog1.0.mp4 | caut_auto | 1039 | fog:1.0 | success | 491 | 5.66 |
| caut_c0__seed1039__fog1.0.mp4 | caut_c0 | 1039 | fog:1.0 | crash | 318 | 6.89 |
| ctrl_nocaut__seed1039__fog1.0.mp4 | ctrl_nocaut | 1039 | fog:1.0 | success | 378 | 7.05 |
| ov_split__seed1039__fog1.0.mp4 | ov_split | 1039 | fog:1.0 | crash | 317 | 6.90 |
| repo_bc_sparse__seed1039__fog1.0.mp4 | repo_bc_sparse | 1039 | fog:1.0 | off_road | 241 | 6.66 |
| base_a__seed1020__rain1.0.mp4 | base_a | 1020 | rain:1.0 | success | 299 | 6.82 |
| caut_auto__seed1020__rain1.0.mp4 | caut_auto | 1020 | rain:1.0 | success | 427 | 5.08 |
| caut_c0__seed1020__rain1.0.mp4 | caut_c0 | 1020 | rain:1.0 | success | 298 | 6.81 |
| ctrl_nocaut__seed1020__rain1.0.mp4 | ctrl_nocaut | 1020 | rain:1.0 | success | 300 | 6.81 |
| ov_split__seed1020__rain1.0.mp4 | ov_split | 1020 | rain:1.0 | success | 313 | 6.21 |
| repo_bc_sparse__seed1020__rain1.0.mp4 | repo_bc_sparse | 1020 | rain:1.0 | off_road | 151 | 5.90 |
| base_a__seed1039__rain1.0.mp4 | base_a | 1039 | rain:1.0 | crash | 318 | 6.87 |
| caut_auto__seed1039__rain1.0.mp4 | caut_auto | 1039 | rain:1.0 | success | 544 | 5.17 |
| caut_c0__seed1039__rain1.0.mp4 | caut_c0 | 1039 | rain:1.0 | crash | 317 | 6.88 |
| ctrl_nocaut__seed1039__rain1.0.mp4 | ctrl_nocaut | 1039 | rain:1.0 | success | 377 | 7.05 |
| ov_split__seed1039__rain1.0.mp4 | ov_split | 1039 | rain:1.0 | crash | 320 | 6.91 |
| repo_bc_sparse__seed1039__rain1.0.mp4 | repo_bc_sparse | 1039 | rain:1.0 | off_road | 241 | 6.66 |

Notes: caut_auto is identical to caut_c0 in clear weather (caution c = 0), so it is shown only in fog/rain. Rule-based references (stock_idm, safe_teacher) see ground truth. Ablation models' input configuration is inferred from the model catalogue.
