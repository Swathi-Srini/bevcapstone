#!/bin/bash
# SPARSE traffic (density 0.05), seeds 5000-5199: always-cautious control (caution forced to 1, every weather incl. clear) at the 800-step and 1600-step limits.
# Compared later with the existing ov_split / caut_auto cells. Post-hoc control, labelled as such. Flag logs/sparse_always_cautious_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for W in clear fog_1.0 rain_0.6 rain_1.0; do
  $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --density 0.05 --arms caut_c1 --only-weather $W --no-reuse --horizon 800 --out bev_rl/adaptive/results/sparse_always_cautious/h800 > logs/sparse_always_cautious_h800_$W.log 2>&1 &
  $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --density 0.05 --arms caut_c1 --only-weather $W --no-reuse --horizon 1600 --out bev_rl/adaptive/results/sparse_always_cautious/h1600 > logs/sparse_always_cautious_h1600_$W.log 2>&1 &
done
wait
touch logs/sparse_always_cautious_done.flag
