#!/bin/bash
# Model B pipeline driver (ONE camera worker; memory-limited).  Created 2026-11-16 under the user's execution authorisation.
# Round 0 is already running (24-list) -> stopped by exact PID after 12 completed episodes; rounds 1-3 collect 12 episodes each; train after every round.
# Then: estimator fit (DAgger whole-route features + 28 preliminary episodes), held-out streams 9041-9042, paired evaluation A/B/C.
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=. SEV_EPISODES=12
M=bev_rl.adaptive.camera_dagger_severity; R=bev_rl/adaptive/results/camera_dagger_severity/e2e_sevB; L=logs/sevB_pipeline.log
log() { echo "$(date '+%F %T') $*" >> $L; }
log "driver started"
# --- round 0: wait for 12 episodes, then stop the round-0 collector by exact PID
until [ "$(ls $R/round0/*.npz 2>/dev/null | wc -l)" -ge 12 ]; do sleep 30; done
powershell -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger_severity collect --round 0" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }'
sleep 8; log "round 0: 12 episodes done, round-0 collector (launcher + worker) stopped"
CNT=$(powershell -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "camera_dagger_severity" -and $_.Name -match "python" -and $_.CommandLine -notmatch "Get-Cim" }).Count' | tr -d '\r'); [ "$CNT" != "0" ] && { log "ABORT: collector still alive ($CNT)"; exit 6; }
for r in 0 1 2 3; do
  if [ $r -gt 0 ]; then log "round $r collect"; $PY -m $M collect --round $r --worker 1 >> logs/sevB_collect_r${r}_w1.log 2>&1; fi
  n=$(ls $R/round$r/*.npz 2>/dev/null | wc -l); log "round $r collected $n episodes; training"
  [ "$n" -lt 8 ] && { log "ABORT: round $r has only $n episodes"; exit 3; }
  $PY -m $M train --round $r >> logs/sevB_train.log 2>&1 || { log "ABORT: train round $r failed"; exit 4; }
  log "round $r trained"
done
cp -n $R/iter_3.pt $R/best.pt && echo '{"round": 3, "selection": "final DAgger round (no validation-based selection; schedule decision)"}' > $R/best.json; log "best.pt = iter_3.pt"
mkdir -p bev_rl/adaptive/results/severity_estimator
$PY -m bev_rl.adaptive.severity_estimator fit --out bev_rl/adaptive/results/severity_estimator/estimator.json >> logs/sevB_pipeline.log 2>&1 || { log "ABORT: estimator fit failed"; exit 5; }
log "estimator fitted"
$PY -m bev_rl.adaptive.measure_retention_camera collect --worker 1 --out bev_rl/adaptive/results/retention_camera --seeds "9041-9042" >> logs/severity_heldout_w1.log 2>&1; log "held-out 9041-9042 done"
# paired evaluation, seed-major so that results accrue as complete A/B/C triples for all conditions
TASKS=""; for s in $(seq 5200 5219); do TASKS="$TASKS;clear:0.0@$s;fog:1.0@$s;rain:1.0@$s"; done; TASKS=${TASKS:1}
log "evaluation start"; $PY -m bev_rl.adaptive.evaluate_severity_policies --out bev_rl/adaptive/results/severity_eval --tasks "$TASKS" --arms A,B,C --worker 1 >> logs/severity_eval_w1.log 2>&1
log "evaluation finished"
