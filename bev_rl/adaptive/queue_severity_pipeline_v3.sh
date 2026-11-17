#!/bin/bash
# Pipeline driver v3 (2026-11-16, user-approved).  Replaces ONLY the driver script v2 (stopped by exact PID 23908; the running A/C evaluation workers are NOT touched).
#  1. wait until the A/C evaluation on seeds 5200-5219 has all 120 episode JSONs (the two running workers finish them);
#  2. snapshot those 120 JSONs to results/severity_eval_20seed_snapshot/ with a SHA-256 manifest, verify it against the originals (originals are only read);
#  3. A/C EXTENSION on seeds 5220-5239, same configuration, separate output dir results/severity_eval_ext, at most two camera workers in total
#     (an extension worker starts as soon as one of the original workers has exited);
#  4. NO B reference pass (needs the user's explicit approval).
# Resumable: every episode is its own JSON and claims are per-episode; re-running this script skips finished episodes.
export PATH=/usr/bin:/bin:/mingw64/bin:$PATH
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
L=logs/sevB_pipeline.log; log() { echo "$(date '+%F %T') $*" >> $L; }
ROOTD=$(pwd); RES=bev_rl/adaptive/results; ORIG=$RES/severity_eval; SNAP=$RES/severity_eval_20seed_snapshot; EXT=$RES/severity_eval_ext
# number of running evaluation launcher processes (the venv launcher stub; one per worker)
nwork() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'evaluate_severity_policies' -and \$_.CommandLine -match 'venv' -and \$_.Name -match 'python' -and \$_.CommandLine -notmatch 'Get-Cim' }).Count" | tr -d '\r'; }
nfiles() { ls $1/episodes/*.json 2>/dev/null | wc -l; }
log "driver v3 started (original A/C workers keep running)"
TASKS=""; for s in $(seq 5220 5239); do TASKS="$TASKS;clear:0.0@$s;fog:1.0@$s;rain:1.0@$s"; done; TASKS=${TASKS:1}
started=0; snapped=0
while true; do
  n=$(nfiles $ORIG)
  if [ "$snapped" = 0 ] && [ "$n" -ge 120 ]; then
    mkdir -p $SNAP/episodes
    for f in $ORIG/episodes/*.json; do cp -n "$f" $SNAP/episodes/; done
    ( cd $SNAP/episodes && sha256sum *.json > ../MANIFEST_sha256.txt )
    ok=1; ( cd $ORIG/episodes && sha256sum -c "$ROOTD/$SNAP/MANIFEST_sha256.txt" >/dev/null 2>&1 ) || ok=0
    cnt=$(wc -l < $SNAP/MANIFEST_sha256.txt)
    if [ "$ok" = 1 ] && [ "$cnt" -eq 120 ]; then snapped=1; echo "{\"created\": \"$(date '+%F %T')\", \"episodes\": $cnt, \"seeds\": \"5200-5219\", \"arms\": \"A,C\", \"note\": \"verified against originals; originals untouched\"}" > $SNAP/SNAPSHOT_INFO.json; log "20-seed snapshot written and VERIFIED ($cnt episodes) -> $SNAP"
    else log "ABORT: snapshot verification failed (ok=$ok, count=$cnt)"; exit 7; fi
  fi
  w=$(nwork)
  if [ "$started" -lt 2 ] && [ "$w" -lt 2 ]; then
    started=$((started + 1)); mkdir -p $EXT
    log "extension worker $started starts (original workers alive: $w)"
    $PY -m bev_rl.adaptive.evaluate_severity_policies --out $EXT --tasks "$TASKS" --arms A,C --worker $started >> logs/severity_eval_ext_w$started.log 2>&1 &
    sleep 60; continue
  fi
  [ "$snapped" = 1 ] && [ "$started" -ge 2 ] && break
  sleep 60
done
wait; log "A/C extension (seeds 5220-5239) finished; B reference pass NOT run (needs user approval)"
