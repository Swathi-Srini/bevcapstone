#!/bin/bash
# RETENTION PASS for the Live Caution camera model (NEW; NOT queued or running until the user launches it).  Usage:  bash queue_retention_camera.sh [WORKERS]     WORKERS default 1, maximum 3.
#   Teacher A drives clear episodes (density 0.05, action noise 0.0); retention frames go through the full camera pipeline in 11 conditions; sensor streams scope S-B-stride2 (held-out weather at even steps, see measure_retention_camera.py).
#   30 training episodes (seeds 9000-9029) + 10 held-out episodes (9040-9049).  Reserve seeds 9030-9039 are NEVER started by this script.
# Output: bev_rl/adaptive/results/retention_camera (separate from retention_camera_timing_check).  Flag: logs/retention_camera.done - created ONLY if `validate` passes (all 40 episodes valid).
# Safety gates (all must hold, in this order):
#   1. logs/retention_camera_full_pass.approved exists   (a file ONLY THE USER creates; this script never creates it)
#   2. logs/camera_training_ov.done exists               (OV-Split finished)
#   3. no camera worker process is running               (camera_dagger collect|validate, evaluate_camera_pipeline, measure_retention_camera)
#   4. the pack / students-eval watchers are not running (they would start their own workers)  -> otherwise ABORT, nothing started
#   5. WORKERS > 1 additionally needs logs/retention_camera_multiworker.approved (user-created; also enforced inside measure_retention_camera.py)
# Failure behaviour: a failed episode is recorded (failures/), its claim is marked failed, the worker continues with the next seed; claims of possibly active workers are never released here;
#   incomplete / invalid data -> flag NOT created (INCOMPLETE logged); reruns are safe (completed episodes are never overwritten).
cd /d/bev_Fresh/bevcapstone || exit 1
WORKERS=${1:-1}
case "$WORKERS" in 1|2|3) ;; *) echo "WORKERS must be 1, 2 or 3 (got '$WORKERS')"; exit 2;; esac
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
OUT=bev_rl/adaptive/results/retention_camera; L=logs/retention_camera.log; SEEDS="9000-9029;9040-9049"
mkdir -p logs; echo "$(date '+%F %T') retention queued (workers=$WORKERS)" >> $L
procs() { powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger (collect|validate)|evaluate_camera_pipeline|measure_retention_camera" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r'; }
watchers() { powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "queue_pack\.sh|queue_students_eval\.sh" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r'; }
until [ -f logs/retention_camera_full_pass.approved ]; do sleep 60; done
until [ -f logs/camera_training_ov.done ]; do sleep 60; done
until [ "$(procs)" = "0" ]; do sleep 60; done
if [ "$(watchers)" != "0" ]; then echo "$(date '+%F %T') ABORT: queue_pack.sh / queue_students_eval.sh watcher is running; they would add workers. Nothing started." >> $L; exit 2; fi
if [ "$WORKERS" != "1" ] && [ ! -f logs/retention_camera_multiworker.approved ]; then echo "$(date '+%F %T') ABORT: WORKERS=$WORKERS needs logs/retention_camera_multiworker.approved. Nothing started." >> $L; exit 2; fi
echo "$(date '+%F %T') approved, OV-Split finished, no camera worker, no pack watcher; starting $WORKERS worker(s)" >> $L
mkdir -p $OUT
for W in $(seq 1 $WORKERS); do $PY -m bev_rl.adaptive.measure_retention_camera collect --worker $W --out $OUT --seeds "$SEEDS" >> logs/retention_camera_w$W.log 2>&1 & sleep 20; done
wait
$PY -m bev_rl.adaptive.measure_retention_camera status --out $OUT --seeds "$SEEDS" >> $L 2>&1
if $PY -m bev_rl.adaptive.measure_retention_camera validate --out $OUT --expect "$SEEDS" >> $L 2>&1; then
  touch logs/retention_camera.done; echo "$(date '+%F %T') retention finished and VALID (40/40 episodes, streams, grids, provenance); flag created" >> $L
else
  echo "$(date '+%F %T') retention INCOMPLETE or INVALID (see summary/validation_report.json and failures/); flag NOT created; stale claims are released only with user approval" >> $L
fi
