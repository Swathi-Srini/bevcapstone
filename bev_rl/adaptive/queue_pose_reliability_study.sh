#!/bin/bash
# Waits for the safe-teacher queue (RAM), then: (1) map-only pose-reliability pair, (2) TWO-CUE pair (map overlay + noisy lane-detector scalars), (3) evals.
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe; L=bev_rl/adaptive/logs
while [ ! -f $L/safe_teacher_study_done.flag ]; do sleep 120; done
$PY -m bev_rl.adaptive.pose_reliability_dagger --name pose_cam_conditioned --condition --cam-cue > $L/pose_cam_conditioned.log 2>&1
$PY -m bev_rl.adaptive.pose_reliability_dagger --name pose_cam_unconditioned --cam-cue > $L/pose_cam_unconditioned.log 2>&1
$PY -m bev_rl.adaptive.evaluate_pose_two_cue_policy --arms pose_cam_conditioned pose_cam_unconditioned --n 40 > $L/eval_pose_two_cue.log 2>&1
$PY -m bev_rl.adaptive.pose_reliability_dagger --name pose_aug_conditioned --condition > $L/pose_aug_conditioned.log 2>&1
$PY -m bev_rl.adaptive.pose_reliability_dagger --name pose_aug_unconditioned > $L/pose_aug_unconditioned.log 2>&1
$PY -m bev_rl.adaptive.evaluate_pose_reliability_policy --arms pose_aug_conditioned pose_aug_unconditioned --n 40 > $L/eval_pose_reliability.log 2>&1
echo DONE > $L/pose_reliability_study_done.flag
