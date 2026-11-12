#!/bin/bash
# Waits for the n=200 all-weather queue's done-flag (file test only, no process polling), then evaluates the safe-teacher student
# (runs/sparse_overlay_split_safe_teacher/best.pt) on the same fresh seeds 5000-5199, clear weather, density 0.05.
cd /d/bev_Fresh/bevcapstone || exit 1
while [ ! -f logs/n200_all_weather_done.flag ]; do sleep 120; done
PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -m bev_rl.adaptive.evaluate_n200_all_weather --n 200 --seed0 5000 \
  --arms safe_student --only-weather clear > logs/n200_safe_student.log 2>&1
touch logs/n200_safe_student_done.flag
