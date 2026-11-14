#!/bin/bash
# QUEUED: camera-trained OV-Split student (the model being trained now) on the FULL camera pipeline: fog 1.0 seeds 5100-5109 (10 episodes) + front-camera/BEV videos, seed 5100, clear / fog 1.0 / rain 1.0.
# 12 episodes, density 0.05, horizon 800, frozen builder. Starts when the deliverable pack has finished (logs/camera_pack.done), OV-Split's best.pt exists and no camera worker runs.
# Uses 4 workers only if the pack's health test found 4 workers healthy; otherwise 3. Output: results/camera_eval_students (the Live Caution student is added to the same folder later).
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
OUT=bev_rl/adaptive/results/camera_eval_students; L=logs/camera_students_eval.log; mkdir -p logs "$OUT"; echo "$(date '+%F %T') queued" >> $L
until [ -f logs/camera_pack.done ] && [ -f bev_rl/adaptive/results/camera_dagger_final/e2e_ov_split_final/best.pt ]; do sleep 60; done
until [ "$(powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger (collect|validate)|evaluate_camera_pipeline" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r')" = "0" ]; do sleep 60; done
N=3; grep -q "4 workers HEALTHY" logs/camera_pack.log && N=4
echo "$(date '+%F %T') started with $N workers" >> $L
for W in $(seq 1 $N); do CAM_MAX_WORKERS=$N $PY -m bev_rl.adaptive.evaluate_camera_pipeline --worker $W --out $OUT --arms ov_camera_final --tasks "fog:1.0@5100-5109;clear:0.0@5100;rain:1.0@5100" --video-seeds 5100 >> logs/camera_students_eval_w$W.log 2>&1 & sleep 20; done; wait
touch logs/camera_students_eval.done; echo "$(date '+%F %T') finished" >> $L
