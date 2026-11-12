#!/bin/bash
# Sparse-traffic (density 0.05) caution study. Sequential. Nothing is deleted. Writes logs/followup_sparse_done.flag at the end.
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe; L=bev_rl/adaptive/logs
$PY -m bev_rl.adaptive.caution_dagger --name caut_a > $L/caut_a.log 2>&1
$PY -m bev_rl.adaptive.caution_dagger --name ctrl_nocaut --no-caution > $L/ctrl_nocaut.log 2>&1
for arm in caut_auto caut_c0 ctrl base_ovsplit; do $PY -m bev_rl.adaptive.eval_caution --arm $arm --n 40 --density 0.05 > $L/eval_sparse_$arm.log 2>&1; done
echo ALL_DONE > $L/followup_sparse_done.flag
