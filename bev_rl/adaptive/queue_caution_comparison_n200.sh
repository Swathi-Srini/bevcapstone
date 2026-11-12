#!/bin/bash
# 200-episode comparison across weather of: the caution-aware model (caution derived from weather), the proposed model, and the two baseline models.
# Same harness and seeds as the existing results (evaluate_n200_all_weather.py): seeds 5000-5199, traffic density 0.05, horizon 800, weather = detector dropout at the
# measured retention, identical dropout stream per episode. Records success, collision, left road, timeout, speed, route completion and estimated energy (Wh/km).
# Already present and reused: proposed model (ov_split, 4 cells), baseline trained with traffic (repo_bc_sparse, 4 cells), baseline trained without traffic (clear cell).
# To run: caution-aware model with caution 0 in clear weather (its clear cell; caution-from-weather is identical there), then with caution from weather in fog 1.0, rain 0.6, rain 1.0;
#         baseline trained without traffic in fog 1.0, rain 0.6, rain 1.0.
# Two parallel streams (one MetaDrive process each, about 3.5 GB). Logs: logs/n200_caution_comparison_{A,B}.log ; flag logs/n200_caution_comparison_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
( $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --arms caut_c0 --only-weather clear \
  && $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --arms caut_auto ) > logs/n200_caution_comparison_A.log 2>&1 &
( $PY -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 --arms repo_bc_clear ) > logs/n200_caution_comparison_B.log 2>&1 &
wait
touch logs/n200_caution_comparison_done.flag
