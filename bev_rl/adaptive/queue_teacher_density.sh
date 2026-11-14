#!/bin/bash
# Teacher results per traffic density (simulator only, no camera). ONE sequential process. Output: results/teacher_density (new folder; nothing existing is written).
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
O=bev_rl/adaptive/results/teacher_density
for D in 0.1 0.2 0.3 0.05; do
  for CELL in "stock_idm clear:0.0" "safe_g4.0 clear:0.0" "cs0_g4.0 fog:1.0" "cs0_g4.0 rain:0.6" "cs0_g4.0 rain:1.0"; do
    set -- $CELL
    $PY -m bev_rl.adaptive.teacher_eval --teacher $1 --weather $2 --density $D --seeds 3100-3149 --horizon 1600 --out $O >> logs/teacher_density.log 2>&1
  done
done
touch logs/teacher_density.done
