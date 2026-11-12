#!/bin/bash
# Sequential PPO reward-combination sweep (RAM-limited: one combo at a time, 2 workers). Start = ov_split + caution input.
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe
run() { name=$1; shift; rm -rf bev_rl/adaptive/runs/$name; $PY -m bev_rl.adaptive.ppo_caution --name $name "$@" > bev_rl/adaptive/logs/$name.log 2>&1; }
run rl_full   --lambda-e 0.05 --safe-speed --w-head 0.5 --crash-c 1.0
run rl_plain
run rl_energy --lambda-e 0.05
run rl_safe   --safe-speed --crash-c 1.0
run rl_head   --w-head 0.5
