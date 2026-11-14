#!/bin/bash
# Queue entry: full camera-pipeline evaluation of the ORIGINAL OV-Split and caution checkpoints (80 episodes: 10 seeds x 4 weathers x 2 models).
# usage: queue_camera_eval.sh <trigger flag file> [workers]
# Waits (negligible memory) until the trigger flag file exists AND no camera collect/validate worker is running, then starts the workers.
cd /d/bev_Fresh/bevcapstone || exit 1
TRIG=$1; N=${2:-3}; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
mkdir -p logs; L=logs/camera_eval_originals.log
echo "$(date '+%F %T') queued, trigger=$TRIG workers=$N" >> $L
until [ -f "$TRIG" ]; do sleep 60; done
while [ "$(powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger (collect|validate)|evaluate_camera_pipeline" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r')" != "0" ]; do sleep 60; done
echo "$(date '+%F %T') started" >> $L
for W in $(seq 1 $N); do CAM_MAX_WORKERS=$N $PY -m bev_rl.adaptive.evaluate_camera_pipeline --worker $W --out bev_rl/adaptive/results/camera_eval_originals --tasks "clear:0.0@5100-5109;fog:1.0@5100-5109;rain:0.6@5100-5109;rain:1.0@5100-5109" >> logs/camera_eval_originals_w$W.log 2>&1 & sleep 20; done; wait
touch logs/camera_eval_originals.done; echo "$(date '+%F %T') finished" >> $L
