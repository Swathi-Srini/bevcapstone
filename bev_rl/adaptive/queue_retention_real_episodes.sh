#!/bin/bash
# Real-episode detector retention at three traffic densities, 30 fresh scenes each (seeds 8000-8029), one burst of 2 frames every 20 steps. Flag: logs/retention_real_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy
for D in 0.05 0.15 0.3; do
  $PY -m bev_rl.adaptive.measure_retention_real_episodes --density $D --seeds 8000-8029 --every 20 --max-steps 400 --out bev_rl/adaptive/results/retention_real_d$D.json > logs/retention_real_d$D.log 2>&1 &
done
wait
touch logs/retention_real_done.flag
