# What has been done so far

Earlier stages (imitation baselines, the five-model comparison, four unsuccessful PPO attempts, the camera pipeline base models, the retention-based caution plan) are documented in the top-level `README.md`, `docs/STATUS_INDEX.md` and `report/`. This file covers the phase 3 work of 15 to 17 November 2026.

| Date | Step |
|---|---|
| 14 Nov | camera-pipeline base models merged (PR #14); retention-based live-caution plan |
| 15 Nov | live-caution pre-registration and queue; retention measurement scripts |
| 16 Nov | **specification change:** the caution must come from an image-based severity estimate; retention caution retired; pre-registration amendment written (`PREREGISTRATION_AMENDMENT_severity_2026-11-16.md`) |
| 16 Nov | Model B (oracle caution) trained by camera-pipeline DAgger: 4 rounds x 12 episodes, density 0.05, horizon 600 (`camera_dagger_severity.py`) |
| 16-17 Nov | teacher validated with the identity caution (50 episodes per condition, privileged: success 49/50 clear; fog and rain 48/50 at 0.6 and 49/50 at 1.0) |
| 16-17 Nov | severity estimator fitted on 25,408 frames and evaluated on held-out routes (seeds 9040-9042) |
| 16-17 Nov | paired A / C evaluation on seeds 5200-5219 (clear, fog 1.0, rain 1.0; 120 episodes; horizon 1600; density 0.05) completed |
| 17 Nov | snapshot of the 20-seed results with SHA-256 manifest, verified; analysis, tables, figures generated; extension (seeds 5220-5239) started |
| 17 Nov | clear-weather failures of C analysed from the episode traces (cause undetermined) |
| 17 Nov | IEEE paper draft, structure-only skeleton, tall architecture diagram with module expansions, Overleaf package; capture script for depth / detection / BEV stills |

## Principles kept throughout
No change to Model A, the estimator, the seeds, the weather conditions or the evaluation configuration after the results existed; completed episode files are never rewritten; evaluation is resumable (claim files); the energy module and traffic density were not changed; the old five-model table was not merged into the new comparison; no results were deleted. Every change is recorded in `bev_rl/adaptive/CHANGELOG_severity_2026-11-16.md` and `..._2026-11-17.md`.
