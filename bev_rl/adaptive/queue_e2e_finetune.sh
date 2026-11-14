#!/bin/bash
# FULL-PIPELINE (camera) plan, run unattended, in this order:
#   0. the grid comparison (compare_grids.py, seeds 5000-5019 requested) is cut at seed 5009; as each weather finishes its CPU is handed to a data-collection worker
#   1. round 0: collect camera-grid episodes (student = ov_split, beta 0.5), 3 workers (densities 0.05 / 0.1 / 0.2), ~3 h
#   2. fine-tune ov_split on them  -> runs/e2e_ov_split_r0
#   3. round 1: collect with the round-0 model (beta 0.25), ~3 h
#   4. fine-tune ov_split on rounds 0+1 -> runs/e2e_ov_split_r1   (the FINAL model; no validation-based selection)
#   5. evaluate original and fine-tuned models through the full pipeline (evaluate_e2e.py, seeds 5000+, priority queue, 3 workers) until stopped
# Flags: logs/e2e_round0_done.flag, logs/e2e_round1_done.flag; log: logs/e2e_finetune_orchestrator.log
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8 SDL_VIDEODRIVER=dummy PYTHONPATH=.
OUT=bev_rl/adaptive/results/e2e_dagger
log() { echo "$(date '+%F %T') $*" >> logs/e2e_finetune_orchestrator.log; }

kill_match() { powershell -Command "Get-CimInstance Win32_Process | Where-Object { \$_.Name -like 'python*' -and \$_.CommandLine -match '$1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"; }

# --- step 0/1: hand each comparison worker's CPU to a round-0 collector as soon as it has done seeds 5000-5009
i=1
for pair in "clear:clear:0.05" "fog:fog:0.1" "rain:rain:0.2"; do
  IFS=: read name kind dens <<< "$pair"
  (
    while ! grep -q "^5009 " logs/grid_compare_${name}.log 2>/dev/null; do sleep 30; done
    kill_match "compare_gri[d]s --weather ${kind}:"
    log "comparison ${name} reached seed 5009, stopped; starting round-0 collector density ${dens}"
    $PY -m bev_rl.adaptive.e2e_dagger collect --round 0 --worker $i --density $dens --ckpt bev_rl/adaptive/runs/ov_split/best.pt --beta 0.5 --hours 3 --out $OUT > logs/e2e_collect_r0_w$i.log 2>&1
  ) &
  i=$((i+1)); sleep 1
done
wait
log "round 0 collection finished"; touch logs/e2e_round0_done.flag
$PY -m bev_rl.adaptive.e2e_dagger train --rounds 0 --init bev_rl/adaptive/runs/ov_split/best.pt --name e2e_ov_split_r0 --out $OUT > logs/e2e_train_r0.log 2>&1
log "trained e2e_ov_split_r0"

i=1
for dens in 0.05 0.1 0.2; do
  $PY -m bev_rl.adaptive.e2e_dagger collect --round 1 --worker $i --density $dens --ckpt bev_rl/adaptive/runs/e2e_ov_split_r0/best.pt --beta 0.25 --hours 3 --out $OUT > logs/e2e_collect_r1_w$i.log 2>&1 &
  i=$((i+1)); sleep 20
done
wait
log "round 1 collection finished"; touch logs/e2e_round1_done.flag
$PY -m bev_rl.adaptive.e2e_dagger train --rounds 0 1 --init bev_rl/adaptive/runs/ov_split/best.pt --name e2e_ov_split_r1 --out $OUT > logs/e2e_train_r1.log 2>&1
log "trained e2e_ov_split_r1 (final)"

CELLS="ov_split:clear:0.0 e2e_ov_split_r1:clear:0.0 ov_split:rain:1.0 e2e_ov_split_r1:rain:1.0 repo_bc_sparse:clear:0.0 repo_bc_sparse:rain:1.0 repo_bc_clear:clear:0.0 repo_bc_clear:rain:1.0 ov_split:fog:1.0 e2e_ov_split_r1:fog:1.0 caut_auto:clear:0.0 caut_auto:rain:1.0"
rm -f bev_rl/adaptive/results/e2e/claims/*; mkdir -p bev_rl/adaptive/results/e2e/claims
for W in 1 2 3; do
  $PY -m bev_rl.adaptive.evaluate_e2e --worker $W --out bev_rl/adaptive/results/e2e --seeds 5000-5039 --block 5 --cells $CELLS > logs/e2e_eval_worker_$W.log 2>&1 &
  sleep 20
done
log "evaluation started"
wait
touch logs/e2e_eval_done.flag; log "evaluation finished"
