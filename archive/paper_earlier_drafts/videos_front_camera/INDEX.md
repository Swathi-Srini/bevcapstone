# Front-camera videos, seed 1039, every model (plain format of bev_policy/outputs/videos/fog_seed1000.mp4: raw 1200x900 front-left frame + weather augmentation, 10 fps real time, one frame per step, no overlays)

The policies do NOT see these pixels: they drive from the simulator-derived BEV (weather = detector dropout, no weather physics). Outcomes are listed here because the videos carry no text.

| file | model | weather | outcome | steps | mean speed m/s |
|---|---|---|---|---|---|
| arch_bev_only__seed1039__clear.mp4 | arch_bev_only | clear:0.0 | off_road | 87 | 2.00 |
| arch_scalar_only__seed1039__clear.mp4 | arch_scalar_only | clear:0.0 | crash | 314 | 6.95 |
| base_a__seed1039__clear.mp4 | base_a | clear:0.0 | crash | 318 | 6.87 |
| caut_c0__seed1039__clear.mp4 | caut_c0 | clear:0.0 | crash | 318 | 6.88 |
| ctrl_nocaut__seed1039__clear.mp4 | ctrl_nocaut | clear:0.0 | success | 377 | 7.05 |
| ov_bev_only__seed1039__clear.mp4 | ov_bev_only | clear:0.0 | off_road | 222 | 7.74 |
| ov_both__seed1039__clear.mp4 | ov_both | clear:0.0 | crash | 320 | 6.84 |
| ov_split__seed1039__clear.mp4 | ov_split | clear:0.0 | crash | 318 | 6.90 |
| repo_bc_clear__seed1039__clear.mp4 | repo_bc_clear | clear:0.0 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__clear.mp4 | repo_bc_sparse | clear:0.0 | off_road | 241 | 6.66 |
| safe_student__seed1039__clear.mp4 | safe_student | clear:0.0 | crash | 320 | 6.89 |
| safe_teacher__seed1039__clear.mp4 | safe_teacher | clear:0.0 | success | 417 | 6.97 |
| stock_idm__seed1039__clear.mp4 | stock_idm | clear:0.0 | crash | 319 | 6.90 |
| arch_bev_only__seed1039__fog0.5.mp4 | arch_bev_only | fog:0.5 | off_road | 87 | 2.00 |
| arch_scalar_only__seed1039__fog0.5.mp4 | arch_scalar_only | fog:0.5 | crash | 314 | 6.95 |
| base_a__seed1039__fog0.5.mp4 | base_a | fog:0.5 | crash | 318 | 6.87 |
| caut_auto__seed1039__fog0.5.mp4 | caut_auto | fog:0.5 | crash | 317 | 6.88 |
| caut_c0__seed1039__fog0.5.mp4 | caut_c0 | fog:0.5 | crash | 317 | 6.88 |
| ctrl_nocaut__seed1039__fog0.5.mp4 | ctrl_nocaut | fog:0.5 | success | 377 | 7.05 |
| ov_bev_only__seed1039__fog0.5.mp4 | ov_bev_only | fog:0.5 | off_road | 238 | 7.75 |
| ov_both__seed1039__fog0.5.mp4 | ov_both | fog:0.5 | crash | 320 | 6.84 |
| ov_split__seed1039__fog0.5.mp4 | ov_split | fog:0.5 | crash | 317 | 6.90 |
| repo_bc_clear__seed1039__fog0.5.mp4 | repo_bc_clear | fog:0.5 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__fog0.5.mp4 | repo_bc_sparse | fog:0.5 | off_road | 241 | 6.66 |
| safe_student__seed1039__fog0.5.mp4 | safe_student | fog:0.5 | crash | 318 | 6.88 |
| safe_teacher__seed1039__fog0.5.mp4 | safe_teacher | fog:0.5 | success | 417 | 6.97 |
| stock_idm__seed1039__fog0.5.mp4 | stock_idm | fog:0.5 | crash | 319 | 6.90 |
| arch_bev_only__seed1039__fog1.0.mp4 | arch_bev_only | fog:1.0 | off_road | 87 | 2.00 |
| arch_scalar_only__seed1039__fog1.0.mp4 | arch_scalar_only | fog:1.0 | crash | 314 | 6.95 |
| base_a__seed1039__fog1.0.mp4 | base_a | fog:1.0 | crash | 318 | 6.87 |
| caut_auto__seed1039__fog1.0.mp4 | caut_auto | fog:1.0 | success | 491 | 5.66 |
| caut_c0__seed1039__fog1.0.mp4 | caut_c0 | fog:1.0 | crash | 317 | 6.88 |
| ctrl_nocaut__seed1039__fog1.0.mp4 | ctrl_nocaut | fog:1.0 | success | 377 | 7.05 |
| ov_bev_only__seed1039__fog1.0.mp4 | ov_bev_only | fog:1.0 | off_road | 219 | 7.89 |
| ov_both__seed1039__fog1.0.mp4 | ov_both | fog:1.0 | crash | 320 | 6.84 |
| ov_split__seed1039__fog1.0.mp4 | ov_split | fog:1.0 | crash | 318 | 6.90 |
| repo_bc_clear__seed1039__fog1.0.mp4 | repo_bc_clear | fog:1.0 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__fog1.0.mp4 | repo_bc_sparse | fog:1.0 | off_road | 241 | 6.66 |
| safe_student__seed1039__fog1.0.mp4 | safe_student | fog:1.0 | crash | 320 | 6.89 |
| safe_teacher__seed1039__fog1.0.mp4 | safe_teacher | fog:1.0 | success | 417 | 6.97 |
| stock_idm__seed1039__fog1.0.mp4 | stock_idm | fog:1.0 | crash | 319 | 6.90 |
| arch_bev_only__seed1039__rain0.5.mp4 | arch_bev_only | rain:0.5 | off_road | 87 | 2.00 |
| arch_scalar_only__seed1039__rain0.5.mp4 | arch_scalar_only | rain:0.5 | crash | 314 | 6.95 |
| base_a__seed1039__rain0.5.mp4 | base_a | rain:0.5 | crash | 318 | 6.87 |
| caut_auto__seed1039__rain0.5.mp4 | caut_auto | rain:0.5 | success | 468 | 5.89 |
| caut_c0__seed1039__rain0.5.mp4 | caut_c0 | rain:0.5 | success | 378 | 7.06 |
| ctrl_nocaut__seed1039__rain0.5.mp4 | ctrl_nocaut | rain:0.5 | success | 378 | 7.05 |
| ov_bev_only__seed1039__rain0.5.mp4 | ov_bev_only | rain:0.5 | off_road | 216 | 7.84 |
| ov_both__seed1039__rain0.5.mp4 | ov_both | rain:0.5 | crash | 320 | 6.84 |
| ov_split__seed1039__rain0.5.mp4 | ov_split | rain:0.5 | crash | 317 | 6.90 |
| repo_bc_clear__seed1039__rain0.5.mp4 | repo_bc_clear | rain:0.5 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__rain0.5.mp4 | repo_bc_sparse | rain:0.5 | off_road | 241 | 6.66 |
| safe_student__seed1039__rain0.5.mp4 | safe_student | rain:0.5 | crash | 320 | 6.89 |
| safe_teacher__seed1039__rain0.5.mp4 | safe_teacher | rain:0.5 | success | 417 | 6.97 |
| stock_idm__seed1039__rain0.5.mp4 | stock_idm | rain:0.5 | crash | 319 | 6.90 |
| arch_bev_only__seed1039__rain1.0.mp4 | arch_bev_only | rain:1.0 | off_road | 87 | 2.00 |
| arch_scalar_only__seed1039__rain1.0.mp4 | arch_scalar_only | rain:1.0 | crash | 314 | 6.95 |
| base_a__seed1039__rain1.0.mp4 | base_a | rain:1.0 | crash | 318 | 6.87 |
| caut_auto__seed1039__rain1.0.mp4 | caut_auto | rain:1.0 | success | 544 | 5.17 |
| caut_c0__seed1039__rain1.0.mp4 | caut_c0 | rain:1.0 | crash | 317 | 6.88 |
| ctrl_nocaut__seed1039__rain1.0.mp4 | ctrl_nocaut | rain:1.0 | success | 377 | 7.05 |
| ov_bev_only__seed1039__rain1.0.mp4 | ov_bev_only | rain:1.0 | off_road | 217 | 7.85 |
| ov_both__seed1039__rain1.0.mp4 | ov_both | rain:1.0 | crash | 320 | 6.84 |
| ov_split__seed1039__rain1.0.mp4 | ov_split | rain:1.0 | crash | 317 | 6.90 |
| repo_bc_clear__seed1039__rain1.0.mp4 | repo_bc_clear | rain:1.0 | crash | 314 | 6.99 |
| repo_bc_sparse__seed1039__rain1.0.mp4 | repo_bc_sparse | rain:1.0 | off_road | 241 | 6.66 |
| safe_student__seed1039__rain1.0.mp4 | safe_student | rain:1.0 | success | 377 | 7.05 |
| safe_teacher__seed1039__rain1.0.mp4 | safe_teacher | rain:1.0 | success | 417 | 6.97 |
| stock_idm__seed1039__rain1.0.mp4 | stock_idm | rain:1.0 | crash | 319 | 6.90 |

caut_auto is identical to caut_c0 in clear weather and is shown only in fog/rain. Ablation models' input configuration is inferred from the model catalogue.
