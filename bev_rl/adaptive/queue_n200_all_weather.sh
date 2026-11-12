#!/bin/bash
# n=200 all-weather evaluation of the six presented models (seeds 5000-5199, density 0.05). Strictly sequential, no process polling.
# Resumable: finished cells are skipped, partial cells resume. Priority order so partial results are already useful.
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 \
  --arms ov_split repo_bc_sparse repo_bc_clear ctrl_nocaut caut_c0 caut_auto > logs/n200_all_weather.log 2>&1
touch logs/n200_all_weather_done.flag
