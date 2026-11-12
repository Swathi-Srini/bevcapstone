# Experiment log (bev_rl)

## 2026-11-01 BC baseline, seeds 1000-1009, lazy real pipeline
`python -m bev_rl.evaluate --policy bc ... --out bev_rl/results/eval_bc_1000-1009`. 10/10 success, route 0.988, reward 326, 8.15 m/s, 79.9 Wh/km, rms jerk 2.57. n_unique_bev=1 over 4589 frames => cached-BEV surrogate justified (zero traffic).

## 2026-11-01 Reward audit (seed 2000, horizon 200, cached BEV, 34 audited frames, 0 mismatches)
- v1 FAILED (crawl < stationary): audit confound (BC steering drifts at 2 m/s). Output kept: reward_audit_v1_FAILED_confounded.json.
- v2 FAILED (fast 162 > bc 144): genuine reward flaw. Output kept: reward_audit_v2_fast_beats_bc.json.
- Reward change after v2: progress credit capped at v_cap*dt; overspeed penalty 0.5*((v-12)/2)^2. Audit script changed after v1 (BC steering fed 8 m/s speed input for crawl/fast).
- v3 PASS: bc 144.0 > crawl 36.0 > stationary -40.0; fast -306.6. Single seed, short horizon: sanity check only.
- Cached-BEV step time ~0.2 s (200 steps in 39-47 s incl. audited frames).

## 2026-11-02 BEV v2 pilot validation
tests/test_bev_v2.py 17 pass (incl. FinalWeightsTests loading final weights). validate_bev: seed 1000, 10 frames, clear/fog0.45/fog1.0/rain0.45/rain1.0; no memory stop. Results in bev_state/validation/bev_v2_pilot_seed1000.json; analysis in 06_BEV_AUDIT.md.
