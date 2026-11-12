#!/bin/bash
# Resume the adaptive-driving pipeline after a restart. Precondition: the CORRECTED measurement exists (bev_rl/adaptive/results/depth_error_measurements.json, 936 rows, 2026-11-10 21:32).
# Steps: discard the invalid first-pass outputs -> refit -> sanity check -> mark measure/fit done -> run the main queue (stages with logs/night_*.done are skipped).
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8; ST=logs/night_status.txt
say() { echo "$(date '+%F %T') $*" >> "$ST"; }
say "resume: started after restart"
[ -f bev_rl/adaptive/results/depth_error_measurements.json ] || { say "resume ABORT: corrected measurement file missing"; exit 1; }
rm -f logs/night_all_done.flag logs/night_hurt.done logs/night_dagger_base.done logs/night_ppo_plain.done logs/night_ppo_ada.done logs/night_eval_main.done logs/night_eval_ref.done logs/night_dagger_ada.done logs/night_eval_ada.done
rm -rf bev_rl/adaptive/results/degraded_eval/hurt_n100 bev_rl/adaptive/runs/deg_base bev_rl/adaptive/runs/ppo_plain bev_rl/adaptive/runs/ppo_ada bev_rl/adaptive/runs/deg_ada
$PY -m bev_rl.adaptive.fit_depth_error_model > logs/night_fit_v2.log 2>&1 || { say "resume ABORT: fit failed"; exit 1; }
$PY - > logs/night_sanity_v2.log 2>&1 <<'PYEOF'
import json, sys
import numpy as np
m = json.load(open("bev_rl/adaptive/results/depth_error_model.json"))["cells"]
def w(c, k):
    n = np.array(c["n"], float); p = np.array(c[k], float); return float((n * p).sum() / max(1.0, n.sum()))
c = m["none:0.0"]; pd, pv = w(c, "p_detect"), w(c, "p_valid")
print("clean: p_detect", round(pd, 3), "p_valid", round(pv, 3), "bin counts", c["n"])
sys.exit(0 if (pd >= 0.5 and pv >= 0.5) else 3)
PYEOF
[ $? -eq 0 ] || { say "resume ABORT: sanity failed - see logs/night_sanity_v2.log"; exit 1; }
touch logs/night_measure.done logs/night_fit.done logs/start_adaptive_now.flag
say "resume: model fitted and sane; launching the main queue"
bash bev_rl/adaptive/queue_adaptive_night.sh
