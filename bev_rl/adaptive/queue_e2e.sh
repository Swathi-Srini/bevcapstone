#!/bin/bash
# FULL END-TO-END evaluation (cameras -> weather filter -> stereo depth + YOLO -> grid -> policy). 3 workers pull from one prioritised queue; results per episode in results/e2e/episodes.
# Memory-limited (each worker renders five 1200x900 cameras): 3 workers on this 13.7 GB machine. Stop any time; finished episodes are kept. Resume: run this script again.
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs bev_rl/adaptive/results/e2e/claims bev_rl/adaptive/results/e2e/episodes
rm -f bev_rl/adaptive/results/e2e/claims/*          # stale claims from an earlier run; finished episodes are skipped through their result files
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
for W in 1 2 3; do
  $PY -m bev_rl.adaptive.evaluate_e2e --worker $W --out bev_rl/adaptive/results/e2e --seeds 5000-5059 --block 10 > logs/e2e_worker_$W.log 2>&1 &
  sleep 20
done
wait
touch logs/e2e_done.flag
