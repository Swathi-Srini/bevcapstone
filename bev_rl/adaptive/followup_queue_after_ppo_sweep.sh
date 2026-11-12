#!/bin/bash
# Runs AFTER rl_queue.sh finishes (RAM: one heavy job at a time). Order: caution DAgger + control, then fresh-seed evals.
# Waits on the PPO sweep via PowerShell (no pgrep in Git Bash). Old partial runs are kept, never deleted.
cd /d/bev_Fresh/bevcapstone
PY=./venv/Scripts/python.exe
L=bev_rl/adaptive/logs
sweep_running() { powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'rl_queue.sh|ppo_caution' }) { exit 0 } else { exit 1 }"; }
while sweep_running; do sleep 60; done
$PY -m bev_rl.adaptive.caution_dagger --name caut_a > $L/caut_a.log 2>&1
$PY -m bev_rl.adaptive.caution_dagger --name ctrl_nocaut --no-caution > $L/ctrl_nocaut.log 2>&1
for arm in caut_auto caut_c0 caut_c1 ctrl; do $PY -m bev_rl.adaptive.eval_caution --arm $arm --n 30 > $L/eval_$arm.log 2>&1; done
echo ALL_DONE > $L/followup_done.flag
