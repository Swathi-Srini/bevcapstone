#!/bin/bash
# Strictly sequential (RAM ~3 GB free): no process-polling (the earlier PowerShell self-match bug deadlocked the old queues).
# 1) teacher ceilings d0.05 n=40  2) 200-seed final eval (seeds 5000-5199) of all 6 arms  3) DAgger from SafeTeacher  4) flag -> pose queue starts
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe; L=bev_rl/adaptive/logs
$PY -m bev_rl.adaptive.evaluate_safe_teacher_ceiling --n 40 --seed0 1010 --density 0.05 > $L/safe_teacher_ceiling_sparse.log 2>&1
$PY -m bev_rl.adaptive.expert_ceiling_lane_change_ablation --n 40 --seed0 1010 --density 0.05 --variants stock no_lane_change > $L/stock_idm_ceiling_sparse.log 2>&1
$PY -m bev_rl.adaptive.evaluate_final_200seed_sparse --n 200 --seed0 5000 --density 0.05 > $L/final_200seed.log 2>&1
echo DONE > $L/final_200seed_done.flag
$PY -m bev_rl.adaptive.dagger_base --name sparse_overlay_split_safe_teacher --safe-teacher --overlay --scalar-keep 0,1,5 --densities 0.0,0.05,0.1 --val-densities 0.05 > $L/sparse_overlay_split_safe_teacher.log 2>&1
echo DONE > $L/safe_teacher_study_done.flag
