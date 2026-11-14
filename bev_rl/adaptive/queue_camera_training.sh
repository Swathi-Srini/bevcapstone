#!/bin/bash
# One camera-pipeline DAgger loop (rounds 0-5, 40 episodes per round, density 0.05) for ONE model:  queue_camera_training.sh ov|caut
# Per round: 3 collect workers (claim the round's 40 episodes) -> train -> 3 validate workers (claim the 10-15 validation episodes) -> select.
# Resumable (finished episodes are skipped). Flag: logs/camera_training_<model>.done. Original checkpoints are never written.
M=$1
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
L=logs/camera_training_$M.log
for R in 0 1 2 3 4 5; do
  echo "$(date '+%F %T') $M round $R collect" >> $L
  for W in 1 2 3; do $PY -m bev_rl.adaptive.camera_dagger collect --model $M --round $R --worker $W >> logs/camera_${M}_collect_w$W.log 2>&1 & sleep 20; done; wait
  echo "$(date '+%F %T') $M round $R train" >> $L
  $PY -m bev_rl.adaptive.camera_dagger train --model $M --round $R >> $L 2>&1
  if [ $R -lt 3 ]; then continue; fi          # validation (seeds 3100-3109, 10 per weather) after rounds 3, 4, 5 only (cost); best of these is selected
  echo "$(date '+%F %T') $M round $R validate" >> $L
  for W in 1 2 3; do $PY -m bev_rl.adaptive.camera_dagger validate --model $M --round $R --worker $W >> logs/camera_${M}_val_w$W.log 2>&1 & sleep 20; done; wait
  $PY -m bev_rl.adaptive.camera_dagger select --model $M --round $R >> $L 2>&1
done
touch logs/camera_training_$M.done; echo "$(date '+%F %T') $M finished" >> $L
