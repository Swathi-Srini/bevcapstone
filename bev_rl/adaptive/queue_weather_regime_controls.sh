#!/bin/bash
# POST-HOC controls at density 0.15, seeds 5000-5199: caution forced to 0 (caut_c0) and the model trained with caution fixed at 0 (ctrl_nocaut), 4 weather each.
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for ARM in caut_c0 ctrl_nocaut; do
  for W in clear fog_1.0 rain_0.6 rain_1.0; do
    $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --density 0.15 --arms $ARM --only-weather $W --no-reuse --out bev_rl/adaptive/results/regime_confirm/d0.15 > logs/regime_controls_${ARM}_$W.log 2>&1 &
  done
done
wait
touch logs/regime_controls_done.flag
