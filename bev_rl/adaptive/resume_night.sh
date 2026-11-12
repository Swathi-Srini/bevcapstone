#!/bin/bash
# RESUME the adaptive-driving pipeline after a restart (2026-11-11). Stages with logs/night_<stage>.done are skipped: measure, fit and hurt are DONE and VALID (corrected measurement v2).
# degraded_dagger has no resume, so an unfinished dagger_base / ppo_* / deg_ada run is wiped and restarted from scratch; evaluation cells resume (per-cell files + .partial every 20 episodes).
# DO NOT use resume_after_reboot.sh or queue_after_remeasure.sh any more: they delete the valid 'hurt' result.
cd /d/bev_Fresh/bevcapstone || exit 1
mkdir -p logs; touch logs/start_adaptive_now.flag
for s in dagger_base:deg_base ppo_plain:ppo_plain ppo_ada:ppo_ada dagger_ada:deg_ada; do st=${s%%:*}; run=${s##*:}; [ -f "logs/night_$st.done" ] || rm -rf "bev_rl/adaptive/runs/$run"; done
echo "$(date '+%F %T') resume_night: restarting main queue (done stages skipped)" >> logs/night_status.txt
bash bev_rl/adaptive/queue_adaptive_night.sh
