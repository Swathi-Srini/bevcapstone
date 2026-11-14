#!/bin/bash
# Teacher validation (simulator only, no cameras). Seeds 3100-3199, density 0.05, horizon 1600 (the 80 s result is read off the same episodes: success within 800 steps).
# Usage: queue_teacher_validation.sh <worker 1|2>.  Two disjoint cell lists in priority order; each cell resumes from its .partial file.
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
OUT=bev_rl/adaptive/results/teacher_validation
if [ "$1" = "1" ]; then CELLS="safe:clear:0.0 cs0:rain:1.0 caut_idm:rain:1.0 cs1:rain:1.0 cs0:fog:1.0 caut_idm:fog:1.0 cs1:fog:1.0 safe:rain:1.0"
else CELLS="stock_idm:clear:0.0 cs0:rain:0.6 caut_idm:rain:0.6 cs1:rain:0.6 stock_idm:fog:1.0"; fi
for c in $CELLS; do
  IFS=: read T K L <<< "$c"
  $PY -m bev_rl.adaptive.teacher_eval --teacher $T --weather $K:$L --seeds 3100-3199 --horizon 1600 --density 0.05 --out $OUT >> logs/teacher_validation_w$1.log 2>&1
done
touch logs/teacher_validation_w$1.done
