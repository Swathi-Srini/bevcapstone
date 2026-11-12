#!/bin/bash
# Overnight pipeline for weather-ADAPTIVE driving under measured depth/detection degradation (see paper/depth_degradation/PREREGISTRATION.md).
# Strictly sequential (RAM), flag files only (no process polling). Resumable: a stage with logs/night_<stage>.done is skipped. Status: logs/night_status.txt
# Start condition: the older n=200 queues finished (logs/n200_safe_student_done.flag) OR a manual go: touch logs/start_adaptive_now.flag
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY="./venv/Scripts/python.exe"; export PYTHONIOENCODING=utf-8; ST=logs/night_status.txt
say() { echo "$(date '+%F %T') $*" >> "$ST"; }
say "queued; waiting for logs/n200_safe_student_done.flag or logs/start_adaptive_now.flag"
while [ ! -f logs/n200_safe_student_done.flag ] && [ ! -f logs/start_adaptive_now.flag ]; do sleep 120; done
say "START"
stage() {
  local name=$1; shift
  if [ -f "logs/night_$name.done" ]; then say "skip $name (done)"; return 0; fi
  say "begin $name"; "$@" > "logs/night_$name.log" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then touch "logs/night_$name.done"; say "done  $name"; else say "FAILED $name rc=$rc"; fi
  return $rc
}
# 1-2: measurement of the perception degradation (needed by everything downstream: stop if it fails)
stage measure $PY -m bev_rl.adaptive.measure_depth_error --seeds 7000-7039 --out bev_rl/adaptive/results/depth_error_measurements.json || { say "ABORT: measurement failed"; exit 1; }
stage fit $PY -m bev_rl.adaptive.fit_depth_error_model || { say "ABORT: fit failed"; exit 1; }
# 3: H1 check (pre-registered): does the calibrated degradation hurt the clean-trained ov_split? n=100
stage hurt $PY -m bev_rl.adaptive.evaluate_degraded --arms ov_split --n 100 --seed0 5000 --tag hurt_n100
# 4-6: base under degradation (SafeTeacher labels), then PPO plain vs adaptive from the SAME base
stage dagger_base $PY -m bev_rl.adaptive.degraded_dagger --name deg_base --iters 5
stage ppo_plain $PY -m bev_rl.adaptive.ppo_adaptive --name ppo_plain --base bev_rl/adaptive/runs/deg_base/best.pt --variant plain --steps 300000 --chunk 100000
stage ppo_ada $PY -m bev_rl.adaptive.ppo_adaptive --name ppo_ada --base bev_rl/adaptive/runs/deg_base/best.pt --variant adaptive --steps 300000 --chunk 100000
# 7-8: final evaluation, n=200 fresh seeds, all weather, identical seeds
stage eval_main $PY -m bev_rl.adaptive.evaluate_degraded --arms ppo_ada ppo_plain deg_base ov_split --n 200 --seed0 5000 --tag cal_d0.05
stage eval_ref $PY -m bev_rl.adaptive.evaluate_degraded --arms repo_bc_clear repo_bc_sparse safe_teacher --n 200 --seed0 5000 --tag cal_d0.05
# 9: extra (time permitting): adaptive DAgger vs the non-adaptive deg_base
stage dagger_ada $PY -m bev_rl.adaptive.degraded_dagger --name deg_ada --iters 5 --adaptive
stage eval_ada $PY -m bev_rl.adaptive.evaluate_degraded --arms deg_ada --n 200 --seed0 5000 --tag cal_d0.05
touch logs/night_all_done.flag; say "ALL DONE"
