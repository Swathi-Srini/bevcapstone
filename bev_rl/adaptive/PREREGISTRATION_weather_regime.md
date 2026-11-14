# Pre-registration: finding a regime where weather lowers driving success (written 2026-11-12, before any data)

Question. In sparse traffic (density 0.05) simulated weather (vehicles missing from the grid at the measured detector retention) does not lower the proposed model's success. Is there a traffic density where it does?

Step 1, pilot (this file's queue): seeds 6000-6039, densities 0.15 / 0.25 / 0.35, models ov_split and caut_auto, weather clear / fog 1.0 / rain 1.0, 800 steps. Pilot is only used to CHOOSE the density; n=40 is too small to claim anything.
Selection rule (fixed now): choose the density where ov_split loses the most success from clear to rain 1.0, provided its clear-weather success is at least 0.50 (otherwise driving is too hard to read any weather effect). If no density shows a drop of at least 0.10, report that no regime was found up to density 0.35.

Step 2, confirmatory: seeds 5000-5199, the chosen density, models ov_split, caut_auto, repo_bc_sparse, repo_bc_clear, weather clear / fog 1.0 / rain 0.6 / rain 1.0, 800 steps and also 1600 steps for caut_auto. Primary hypothesis: ov_split success in rain 1.0 is lower than in clear (exact McNemar, two-sided, alpha 0.05). Secondary: caut_auto loses less success than ov_split in weather (paired, same seeds).
All results reported whatever they show.

## Pilot outcome (read 2026-11-12, before the confirmatory run)
Pilot success at n=40 (seeds 6000-6039), clear / fog 1.0 / rain 1.0:
- density 0.15: ov_split 0.550 / 0.475 / 0.400; caut_auto 0.325 / 0.525 / 0.575
- density 0.25: ov_split 0.250 / 0.125 / 0.100; caut_auto 0.100 / 0.175 / 0.225
- density 0.35: ov_split 0.125 / 0.025 / 0.025; caut_auto 0.025 / 0.050 / 0.025
Selection rule applied: only density 0.15 has clear-weather success >= 0.50 for ov_split; drop clear -> rain 1.0 is 0.15 (>= 0.10). Chosen density: 0.15.
Confirmatory run launched with the step-2 design above (4 models x 4 weather, 800 steps, seeds 5000-5199, density 0.15, no reuse of sparse-traffic results).

## Confirmatory outcome (read 2026-11-13) and added controls
Primary hypothesis (ov_split lower in rain 1.0 than in clear, density 0.15, n=200): CONFIRMED. 0.515 -> 0.370 (36 episodes only clear solved, 7 only rain solved).
Secondary (caut_auto better than ov_split in weather): CONFIRMED in fog 1.0, rain 0.6, rain 1.0; but caut_auto is WORSE in clear (0.355 against 0.515).
Added AFTER seeing these results (post-hoc controls, to be labelled as such): caut_c0 (same network, caution forced to 0) and ctrl_nocaut (caution input trained fixed at 0) in all four weather conditions, density 0.15, seeds 5000-5199.
Purpose: separate "the caution input rescues driving" from "any slower driver would do better".

## Control results (read 2026-11-13) and a further control
caut_c0 (same network, caution forced 0), density 0.15, 200 episodes: clear 0.355, fog 1.0 0.360, rain 0.6 0.350, rain 1.0 0.350. ctrl_nocaut: 0.325 / 0.360 / 0.370 / 0.380.
caut_auto minus caut_c0: +0.165 fog, +0.215 rain 0.6, +0.230 rain 1.0 (all p < 0.001). So the caution INPUT is what raises success in weather.
Open question: is the gain about perception loss or simply about driving cautiously? Added control caut_c1: same network, caution forced to 1 in every weather INCLUDING clear. Post-hoc, labelled as such.
