#!/bin/bash
# Re-run ONLY the episodes that hit the 800-step limit, at the 1600-step limit (the others are unaffected by a longer limit).
# Proposed model, clear: seeds [5151]; image-only model, clear: seeds [5029, 5039, 5054, 5076, 5082, 5109, 5150, 5156, 5159, 5170, 5180, 5181, 5190]. Flag: logs/rerun_timeouts_1600_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
$PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 1 --seed0 5151 --density 0.05 --arms ov_split --only-weather clear --no-reuse --horizon 1600 --out bev_rl/adaptive/results/n200_horizon1600/rerun_ov_split_clear_seed5151 > logs/rerun_ov_5151.log 2>&1
(
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5029 --horizon 1600 --conds clear > logs/rerun_vcam_5029.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5039 --horizon 1600 --conds clear > logs/rerun_vcam_5039.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5054 --horizon 1600 --conds clear > logs/rerun_vcam_5054.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5076 --horizon 1600 --conds clear > logs/rerun_vcam_5076.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5082 --horizon 1600 --conds clear > logs/rerun_vcam_5082.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5109 --horizon 1600 --conds clear > logs/rerun_vcam_5109.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5150 --horizon 1600 --conds clear > logs/rerun_vcam_5150.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5156 --horizon 1600 --conds clear > logs/rerun_vcam_5156.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5159 --horizon 1600 --conds clear > logs/rerun_vcam_5159.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5170 --horizon 1600 --conds clear > logs/rerun_vcam_5170.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5180 --horizon 1600 --conds clear > logs/rerun_vcam_5180.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5181 --horizon 1600 --conds clear > logs/rerun_vcam_5181.log 2>&1
$PY -m bev_rl.adaptive.vanilla_camera_cnn eval --name vcam --n 1 --seed0 5190 --horizon 1600 --conds clear > logs/rerun_vcam_5190.log 2>&1
) &
wait
touch logs/rerun_timeouts_1600_done.flag
