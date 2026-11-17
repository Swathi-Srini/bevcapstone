#!/bin/bash
# Pipeline driver v2 (2026-11-16 ~11:30).  Supersedes queue_severity_pipeline.sh from round 2 on (v1 stopped by PID while round 2 was collecting; its round-2 worker 1 keeps running).
# Change of scope (user, 2026-11-16): the FINAL system has only TWO models: A (non-adaptive) and C (adaptive = the oracle-trained network + image-estimated caution).  B is a TRAINING REFERENCE only.
#   -> evaluation order: A and C on all paired seeds first; the B reference pass runs afterwards on as many seeds as time allows (not a separate final model).
# Two camera workers (memory permitting; the held-out collector is a single worker because the retention script's multi-worker flag is user-created only).
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=. SEV_EPISODES=12
M=bev_rl.adaptive.camera_dagger_severity; R=bev_rl/adaptive/results/camera_dagger_severity/e2e_sevB; L=logs/sevB_pipeline.log
log() { echo "$(date '+%F %T') $*" >> $L; }
alive() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '$1' -and \$_.Name -match 'python' -and \$_.CommandLine -notmatch 'Get-Cim' }).Count" | tr -d '\r'; }
log "driver v2 started"
until [ "$(alive 'camera_dagger_severity collect')" = "0" ]; do sleep 30; done      # round-2 workers (started outside this script) finish first
for r in 2 3; do
  if [ $r -gt 2 ]; then
    log "round $r collect (2 workers)"
    for W in 1 2; do $PY -m $M collect --round $r --worker $W >> logs/sevB_collect_r${r}_w$W.log 2>&1 & sleep 20; done; wait
  fi
  n=$(ls $R/round$r/*.npz 2>/dev/null | wc -l); log "round $r collected $n episodes; training"
  [ "$n" -lt 8 ] && { log "ABORT: round $r has only $n episodes"; exit 3; }
  $PY -m $M train --round $r >> logs/sevB_train.log 2>&1 || { log "ABORT: train round $r failed"; exit 4; }
  log "round $r trained"
done
cp -n $R/iter_3.pt $R/best.pt && echo '{"round": 3, "selection": "final DAgger round (no validation-based selection; schedule decision)"}' > $R/best.json; log "best.pt = iter_3.pt"
mkdir -p bev_rl/adaptive/results/severity_estimator
$PY -m bev_rl.adaptive.severity_estimator fit --out bev_rl/adaptive/results/severity_estimator/estimator.json >> logs/sevB_pipeline.log 2>&1 || { log "ABORT: estimator fit failed"; exit 5; }
log "estimator fitted"
OUT=bev_rl/adaptive/results/severity_eval
TASKS=""; for s in $(seq 5200 5219); do TASKS="$TASKS;clear:0.0@$s;fog:1.0@$s;rain:1.0@$s"; done; TASKS=${TASKS:1}
log "held-out 9041-9042 (1 process) + evaluation worker 1 (A,C)"
$PY -m bev_rl.adaptive.measure_retention_camera collect --worker 1 --out bev_rl/adaptive/results/retention_camera --seeds "9041-9042" >> logs/severity_heldout_w1.log 2>&1 &
HO=$!
$PY -m bev_rl.adaptive.evaluate_severity_policies --out $OUT --tasks "$TASKS" --arms A,C --worker 1 >> logs/severity_eval_w1.log 2>&1 &
sleep 20; wait $HO; log "held-out 9041-9042 done; evaluation worker 2 (A,C) starts"
$PY -m bev_rl.adaptive.evaluate_severity_policies --out $OUT --tasks "$TASKS" --arms A,C --worker 2 >> logs/severity_eval_w2.log 2>&1 &
wait; log "A,C evaluation finished"
# B reference pass (training reference only) on the first 12 seeds
BT=""; for s in $(seq 5200 5211); do BT="$BT;clear:0.0@$s;fog:1.0@$s;rain:1.0@$s"; done; BT=${BT:1}
for W in 1 2; do $PY -m bev_rl.adaptive.evaluate_severity_policies --out $OUT --tasks "$BT" --arms B --worker $W >> logs/severity_eval_B_w$W.log 2>&1 & sleep 20; done; wait
log "B reference pass finished"
