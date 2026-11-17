# Change log — severity-based adaptive system (2026-11-16)
Files CREATED (no existing source file, config, preregistration, checkpoint or the energy module was modified):
- bev_rl/adaptive/camera_dagger_severity.py        Model B camera DAgger (wrapper of camera_dagger.py; oracle identity caution; own output dir)
- bev_rl/adaptive/severity_estimator.py            kind classifier + fog/rain regressors, live filter, fit/eval commands
- bev_rl/adaptive/evaluate_severity_policies.py    paired A/B/C evaluation with per-step instrumentation
- bev_rl/adaptive/validate_teacher_identity_caution.py   Teacher B validation with identity caution
- bev_rl/adaptive/PREREGISTRATION_AMENDMENT_severity_2026-11-16.md
- bev_rl/adaptive/results/camera_dagger_severity/FROZEN_MANIFEST.json   (written by `camera_dagger_severity freeze`)
Commands / processes (chronological) are appended below.
- 00:40 held-out stream collection: measure_retention_camera collect --worker 1 --out results/retention_camera --seeds 9040-9049 (PID 26920) log logs/severity_heldout_w1.log
- 00:42 Teacher B validation: validate_teacher_identity_caution for clear, fog 0.6/1.0, rain 0.6/1.0 (log logs/teacher_identity_caution.log, out results/teacher_identity_caution/)
- 00:43 camera_dagger_severity freeze -> results/camera_dagger_severity/FROZEN_MANIFEST.json
- 01:16 held-out worker (PID 26920) stopped by exact PID after seed 9040 completed (2138 s, peak WS 6.1 GB, min system available 1.6 GB); interrupted claim s9041 preserved as s9041.interrupted_by_operator_20261116_0117.json. Seeds 9041-9049 NOT collected yet (each ~36 min; resume later).
- 01:18 Model B DAgger round 0 collection: camera_dagger_severity collect --round 0 --worker 1 (1 worker; memory: committed ~16.8 GB vs 13.7 GB physical, 2nd worker unsafe). log logs/sevB_collect_r0_w1.log
- 01:45 pipeline driver bev_rl/adaptive/queue_severity_pipeline.sh launched (rounds 0-3 @12 ep, train, estimator fit, held-out 9041-9042, eval A/B/C seeds 5200-5219 horizon 1600); log logs/sevB_pipeline.log. Decisions: episodes/round 12 (measured 18.3 min/ep), best=iter_3 (no validation), 1 worker.
- 01:39 driver relaunched (Start-Process, PID 23960; PATH fix; kill-all-matching for the round-0 collector). Round 0 worker: venv launcher PID 22572 + python PID 14604 (started 01:17). First episode: seed 2301 rain 0.6, 600 steps, 1099 s.
- 11:20 SCOPE (user): final system = TWO models, A non-adaptive and C adaptive; B is a training reference only. Evaluation re-ordered: A and C on all paired seeds first, B reference pass afterwards.
- 11:25 v1 driver (PID 23960) stopped (round-2 worker 1 PIDs 24696/18544 left running). Round-2 worker 2 started (launcher 19524; memory 1.45 GB free, pages-in 118/s, CPU ~30%: not thrashing).
- 11:30 queue_severity_pipeline_v2.sh launched (PID 23908): finish round 2, train, round 3 with 2 workers, best=iter_3, estimator fit, held-out 9041-9042 (1 proc) + eval A,C (worker 1, then worker 2), then B reference seeds 5200-5211.
- 11:55 USER CONFIRMED Nov 16 spec governs; no effective-range architecture, no restart. Difference vs drawio v2 / Final_Methodology documented in amendment section 8.

## 19:17 - driver v3 (user-approved: 20-seed snapshot, then A/C extension on seeds 5220-5239)
- Stopped ONLY the v2 driver script, exact PID 23908 (its two subshells 26968/19096 and the two running A/C evaluation workers, python PIDs 13424/14612 and 23900/17292, were not touched).
- New file `queue_severity_pipeline_v3.sh` (PID 5292): waits for 120 episode JSONs in results/severity_eval; copies them to results/severity_eval_20seed_snapshot/ with MANIFEST_sha256.txt; verifies against the originals (originals read-only); then A/C extension, seeds 5220-5239, same configuration, separate output results/severity_eval_ext, <= 2 workers in total (an extension worker starts as soon as an original worker has exited). B reference pass NOT scheduled (needs user approval). Logs: logs/sevB_pipeline.log, logs/severity_eval_ext_w{1,2}.log, logs/sevB_driver_v3.{out,err}.
- No change to models, estimator, seeds, weather, horizon or evaluation code.
