#!/bin/bash
# Chain 2 (2026-11-10): after the CORRECTED measurement (logs/measure_v2.log ends with exit=0): refit -> sanity check -> discard invalid stage outputs -> relaunch the main queue.
# File-content waits only (grep on log/status files), no process polling. Stops with a status line if anything is implausible.
cd /d/bev_Fresh/bevcapstone || exit 1
PY=./venv/Scripts/python.exe; export PYTHONIOENCODING=utf-8; ST=logs/night_status.txt
say() { echo "$(date '+%F %T') $*" >> "$ST"; }
say "chain2: waiting for logs/measure_v2.log exit line"
while ! grep -q "^exit=" logs/measure_v2.log 2>/dev/null; do sleep 60; done
grep -q "^exit=0" logs/measure_v2.log || { say "chain2 ABORT: corrected measurement failed (see logs/measure_v2.log)"; exit 1; }
# let the first queue finish failing its downstream stages (it writes ALL DONE); give up waiting after 40 min
for i in $(seq 1 80); do grep -q "ALL DONE" "$ST" && break; sleep 30; done
say "chain2: fitting corrected model"
$PY -m bev_rl.adaptive.fit_depth_error_model > logs/night_fit_v2.log 2>&1 || { say "chain2 ABORT: fit failed"; exit 1; }
$PY - > logs/night_sanity_v2.log 2>&1 <<'PYEOF'
import json, sys
import numpy as np
m = json.load(open("bev_rl/adaptive/results/depth_error_model.json"))["cells"]
def w(c, k):
    n = np.array(c["n"], float); p = np.array(c[k], float); return float((n * p).sum() / max(1.0, n.sum()))
c = m["none:0.0"]; pd, pv = w(c, "p_detect"), w(c, "p_valid")
print("clean: p_detect", round(pd, 3), "p_valid(box depth | detected)", round(pv, 3), "bin counts", c["n"])
for k in m: print(k, "p_detect", round(w(m[k], "p_detect"), 3), "p_valid", round(w(m[k], "p_valid"), 3))
sys.exit(0 if (pd >= 0.5 and pv >= 0.5) else 3)
PYEOF
if [ $? -ne 0 ]; then say "chain2 ABORT: sanity failed (clean detection or clean valid-depth < 0.5) - see logs/night_sanity_v2.log; NOT training on this model"; exit 1; fi
say "chain2: sanity ok; discarding outputs of the invalid first pass and relaunching the main queue"
rm -f logs/night_all_done.flag logs/night_hurt.done logs/night_dagger_base.done logs/night_ppo_plain.done logs/night_ppo_ada.done logs/night_eval_main.done logs/night_eval_ref.done logs/night_dagger_ada.done logs/night_eval_ada.done
rm -rf bev_rl/adaptive/results/degraded_eval/hurt_n100 bev_rl/adaptive/runs/deg_base bev_rl/adaptive/runs/ppo_plain bev_rl/adaptive/runs/ppo_ada bev_rl/adaptive/runs/deg_ada
touch logs/night_measure.done logs/night_fit.done
bash bev_rl/adaptive/queue_adaptive_night.sh
