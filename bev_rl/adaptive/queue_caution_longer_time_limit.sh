#!/bin/bash
# Caution-aware model re-evaluated with a 1600-step (160 s) limit instead of 800 steps (80 s), in the three weather cells where it timed out.
# Same seeds 5000-5199, density 0.05, same dropout stream. Outputs: results/n200_horizon1600/caut_auto__<weather>.json ; flag logs/n200_horizon1600_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for W in fog_1.0 rain_0.6 rain_1.0; do
  $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --arms caut_auto --only-weather $W --horizon 1600 --no-reuse --out bev_rl/adaptive/results/n200_horizon1600 > logs/n200_horizon1600_$W.log 2>&1 &
done
wait
touch logs/n200_horizon1600_done.flag
