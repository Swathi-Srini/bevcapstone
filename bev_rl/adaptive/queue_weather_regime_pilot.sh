#!/bin/bash
# PILOT to find a traffic density where simulated weather lowers success. Fresh seed block 6000-6039 (never used before), horizon 800.
# Models: ov_split (proposed) and caut_auto (caution-aware). Weather: clear, fog 1.0, rain 1.0. Densities 0.15, 0.25, 0.35.
# Outputs: bev_rl/adaptive/results/regime_pilot/d<density>/ ; flag logs/regime_pilot_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for D in 0.15 0.25 0.35; do
  for ARM in ov_split caut_auto; do
    $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 40 --seed0 6000 --density $D --arms $ARM --only-weather clear fog_1.0 rain_1.0 --no-reuse --out bev_rl/adaptive/results/regime_pilot/d$D > logs/regime_pilot_${ARM}_d$D.log 2>&1 &
  done
done
wait
touch logs/regime_pilot_done.flag
