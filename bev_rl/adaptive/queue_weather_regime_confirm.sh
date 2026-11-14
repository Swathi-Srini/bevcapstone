#!/bin/bash
# CONFIRMATORY n=200 at density 0.15 (chosen by the pilot, see PREREGISTRATION_weather_regime.md). 16 parallel streams = 4 models x 4 weather.
# Outputs: bev_rl/adaptive/results/regime_confirm/d0.15/ ; flag logs/regime_confirm_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for ARM in ov_split caut_auto repo_bc_sparse repo_bc_clear; do
  for W in clear fog_1.0 rain_0.6 rain_1.0; do
    $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --density 0.15 --arms $ARM --only-weather $W --no-reuse --out bev_rl/adaptive/results/regime_confirm/d0.15 > logs/regime_confirm_${ARM}_$W.log 2>&1 &
  done
done
wait
touch logs/regime_confirm_done.flag
