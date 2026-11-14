#!/bin/bash
# PILOT of the camera grid builder: original proposed checkpoint (ov_split, unchanged), 10 episodes (seeds 5000-5009) per weather (clear, fog 1.0, rain 0.6, rain 1.0), density 0.05,
# every step through cameras -> weather filter -> stereo depth + YOLO -> corrected builder. Starts only after the read-only grid comparisons (seeds 5000-5019) have finished.
# 3 workers pull from one queue (claim files). Flag: logs/pilot_camera_builder_done.flag. Results: bev_rl/adaptive/results/pilot_camera_builder
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
while [ $(powershell -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -match 'compare_gri[d]s' } | Measure-Object).Count") -gt 0 ]; do sleep 60; done
echo "$(date '+%F %T') comparisons finished, pilot starting" >> logs/pilot_camera_builder.log
mkdir -p bev_rl/adaptive/results/pilot_camera_builder/claims
for W in 1 2 3; do
  $PY -m bev_rl.adaptive.pilot_camera_builder --worker $W --out bev_rl/adaptive/results/pilot_camera_builder --seeds 5000-5009 --density 0.05 > logs/pilot_camera_builder_w$W.log 2>&1 &
  sleep 30
done
wait
touch logs/pilot_camera_builder_done.flag; echo "$(date '+%F %T') pilot finished" >> logs/pilot_camera_builder.log
