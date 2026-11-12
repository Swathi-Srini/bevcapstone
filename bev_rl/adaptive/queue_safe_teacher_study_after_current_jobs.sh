#!/bin/bash
# Waits until no other training/eval job is running (RAM), then: (1) teacher ceilings at sparse traffic, (2) DAgger students from the SAFE teacher.
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe; L=bev_rl/adaptive/logs
busy() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'caution_dagger|eval_caution|evaluate_localisation|followup_caution' }) { exit 0 } else { exit 1 }"; }
while busy; do sleep 120; done
$PY -m bev_rl.adaptive.evaluate_safe_teacher_ceiling --n 40 --seed0 1010 --density 0.05 > $L/safe_teacher_ceiling_sparse.log 2>&1
$PY -m bev_rl.adaptive.expert_ceiling_lane_change_ablation --n 40 --seed0 1010 --density 0.05 --variants stock no_lane_change > $L/stock_idm_ceiling_sparse.log 2>&1
$PY -m bev_rl.adaptive.dagger_base --name sparse_overlay_split_safe_teacher --safe-teacher --overlay --scalar-keep 0,1,5 --densities 0.0,0.05,0.1 --val-densities 0.05 > $L/sparse_overlay_split_safe_teacher.log 2>&1
echo DONE > $L/safe_teacher_study_done.flag
