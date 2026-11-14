#!/bin/bash
# PILOT v2 (builder with the side-camera reflection fix, fitted geometry, true-size footprint): original ov_split checkpoint, seeds 5000-5009 x {clear, fog 1.0, rain 0.6, rain 1.0}, density 0.05.
# Results: bev_rl/adaptive/results/pilot_camera_builder_v2 (the 13 episodes of builder v1 in results/pilot_camera_builder are kept untouched). Flag: logs/pilot_camera_builder_v2_done.flag
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
mkdir -p bev_rl/adaptive/results/pilot_camera_builder_v2/claims
for W in 1 2 3; do
  $PY -m bev_rl.adaptive.pilot_camera_builder --worker $W --out bev_rl/adaptive/results/pilot_camera_builder_v2 --seeds 5000-5009 --density 0.05 > logs/pilot_camera_builder_v2_w$W.log 2>&1 &
  sleep 30
done
wait
touch logs/pilot_camera_builder_v2_done.flag
