#!/bin/bash
# Caution-SafeTeacher (fixed thresholds, cs0) WITH the curvature guard at 4.0 m/s^2, validation seeds 3100-3199, density 0.05, horizon 1600.
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
for W in rain:1.0 rain:0.6 fog:1.0; do
  $PY -m bev_rl.adaptive.teacher_eval --teacher cs0_g4.0 --weather $W --seeds 3100-3199 --horizon 1600 --density 0.05 --out bev_rl/adaptive/results/teacher_validation >> logs/teacher_validation_guard_caution.log 2>&1
done
touch logs/teacher_validation_guard_caution.done
