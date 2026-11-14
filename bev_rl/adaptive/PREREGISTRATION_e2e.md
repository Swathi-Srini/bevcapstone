# Full-pipeline (camera) evaluation: plan written before any result (2026-11-14)

## Why
The 200-episode results (README of results, `Five_Model_Comparison.pdf`) built the policy's grid from the simulator's vehicle list, with random vehicle dropout in weather. The repository's
own pipeline builds the grid from the cameras (weather filter -> stereo depth + YOLO -> footprints -> grid, `BEVStateAssembler.assemble`). Nothing in the teammates' modules is modified;
`e2e_pipeline.py` only calls them.

## Step 1: are the grids the same? (`compare_grids.py`)
Drive seeds 5000-5009 exactly as the 200-episode test did (proposed model, same dropout stream), sample the camera grid every 10 steps and every 3rd step while a vehicle is in the window,
compare cell by cell with the simulator-list grid (no dropout). Weathers: clear, fog 1.0, rain 1.0. All grids are stored (`results/grid_comparison/*.npz`). Reported: exact-equality counts split
into frames with and without a vehicle in the window, vehicle-cell overlap, vehicles missed and invented.

## Step 2: fine-tune on camera grids (`e2e_dagger.py`)
Dataset aggregation, same recipe as the original training (privileged collision-aware teacher labels, student drives with probability 1-beta, action noise 0.1) but the student's input is the
camera grid. Training seeds 2000-2999 only. Weather per episode: clear (x2), fog 1.0, rain 0.6, rain 1.0. Densities 0.05 / 0.1 / 0.2 (one per worker). Horizon 600.
Round 0: student = proposed model, beta 0.5, ~3 h. Fine-tune (Adam 3e-4, 8 epochs, batch 128). Round 1: student = round-0 model, beta 0.25, ~3 h. Fine-tune again from the proposed model on rounds 0+1.
**The final model is `e2e_ov_split_r1`; no validation-based checkpoint selection, no second try.**

## Step 3: evaluate through the full pipeline (`evaluate_e2e.py`)
Seeds 5000-5039 (never used in training), density 0.05, horizon 800, every step through the camera pipeline. Cells in priority order (the queue is stopped when time runs out and only seeds that
finished for BOTH compared models are used): proposed and fine-tuned in clear and rain 1.0; the two GitHub baselines in clear and rain 1.0; then fog 1.0; then caution-aware (caution supplied from the weather label).
Paired exact McNemar tests, Wilson intervals. Whatever number of episodes finishes is the number reported; it will be far smaller than 200 (about 5000 camera steps per hour on this CPU-only machine).

## What is NOT claimed
That the 200-episode simulator-list results describe the integrated system. They describe ideal perception plus measured vehicle loss.
