#!/bin/bash
# POST-HOC control at density 0.15: caution forced to 1 in all weather, including clear. Seeds 5000-5199. Flag logs/caution_forced_one_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for W in clear fog_1.0 rain_0.6 rain_1.0; do
  $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --density 0.15 --arms caut_c1 --only-weather $W --no-reuse --out bev_rl/adaptive/results/regime_confirm/d0.15 > logs/caution_forced_one_$W.log 2>&1 &
done
wait
touch logs/caution_forced_one_done.flag
