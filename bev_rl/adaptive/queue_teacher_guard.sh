#!/bin/bash
# Curvature-guard variants of the SafeTeacher on the validation seeds (design only; seeds 3100-3199, clear = weather-invariant for a privileged teacher without caution).
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
for T in safe_g3.0 safe_g4.0; do
  $PY -m bev_rl.adaptive.teacher_eval --teacher $T --weather clear:0.0 --seeds 3100-3199 --horizon 1600 --density 0.05 --out bev_rl/adaptive/results/teacher_validation >> logs/teacher_validation_guard.log 2>&1
done
touch logs/teacher_validation_guard.done
