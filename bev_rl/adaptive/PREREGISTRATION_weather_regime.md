# Pre-registration: finding a regime where weather lowers driving success (written 2026-11-12, before any data)

Question. In sparse traffic (density 0.05) simulated weather (vehicles missing from the grid at the measured detector retention) does not lower the proposed model's success. Is there a traffic density where it does?

Step 1, pilot (this file's queue): seeds 6000-6039, densities 0.15 / 0.25 / 0.35, models ov_split and caut_auto, weather clear / fog 1.0 / rain 1.0, 800 steps. Pilot is only used to CHOOSE the density; n=40 is too small to claim anything.
Selection rule (fixed now): choose the density where ov_split loses the most success from clear to rain 1.0, provided its clear-weather success is at least 0.50 (otherwise driving is too hard to read any weather effect). If no density shows a drop of at least 0.10, report that no regime was found up to density 0.35.

Step 2, confirmatory: seeds 5000-5199, the chosen density, models ov_split, caut_auto, repo_bc_sparse, repo_bc_clear, weather clear / fog 1.0 / rain 0.6 / rain 1.0, 800 steps and also 1600 steps for caut_auto. Primary hypothesis: ov_split success in rain 1.0 is lower than in clear (exact McNemar, two-sided, alpha 0.05). Secondary: caut_auto loses less success than ov_split in weather (paired, same seeds).
All results reported whatever they show.
