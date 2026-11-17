# Status (17 November 2026, evening)

## The final system (Nov 16 specification, governs)
camera images -> image-based severity estimate -> caution score -> adaptive driving policy -> safe driving, with energy measured.
Only **two** final models are compared: **A** (non-adaptive OV-Split camera policy, unmodified) and **C** (adaptive: the network trained with the true caution, run with the image-estimated caution at inference). **B** is the training reference only. See `ARCHITECTURE.md`. The effective-range architecture was NOT adopted and training was NOT restarted (pre-registration amendment, section 8).

## Done and verified
- Model B (oracle-caution reference) trained by camera-pipeline DAgger: 4 rounds x 12 episodes, `best.pt = iter_3.pt`.
- Image-based severity estimator fitted (25,408 frames) and evaluated on held-out routes (seeds 9040-9042, 8,427 frames): kind accuracy 0.976, fog MAE 0.045, rain MAE 0.051, clear false caution 0.000.
- Paired A / C evaluation on seeds 5200-5219 in clear, fog 1.0 and rain 1.0 (120 episodes) finished. The episode files are copied to `results/severity_eval_20seed_snapshot/` with a SHA-256 manifest, verified against the originals.
- Analysis, tables and figures generated from the snapshot; paper draft (IEEE) written; architecture diagrams drawn.

## Running when this was written
- **Extension, seeds 5220-5239** (A and C, same configuration, two workers, separate output directory `results/severity_eval_ext`). About a third done at the time of writing; the expected finish is roughly 18 November, late morning to afternoon IST. **Its partial results are not in this branch.** The 20-seed numbers do not depend on it and will not be altered by it.

## Open
| Item | State |
|---|---|
| Cause of the three clear-weather failures of C (two stalls, one collision, estimated caution exactly 0 in all) | undetermined; diagnostic replay (BEV and detections at the stall step) proposed, not run |
| Depth map / detection / BEV screenshots under weather | capture script written (`bev_rl/adaptive/capture_pipeline_stills.py`); one attempt was aborted by a memory guard to protect the running evaluation; to be re-run when memory allows |
| Model B reference evaluation pass | not run; only with explicit approval |
| Paper: author names, title wording, extension subsection, bibliography check | placeholders / pending (`report/paper_severity/`) |
| Phase 3 report chapters (PRS, HLD, LLD, pseudocode, test plan), user manual, plagiarism report (PESU library, limit 15 %) | not written yet |
| Extension results into the tables | after the extension finishes |

## Known limitations (also stated in the paper)
Synthetic image-space weather (the estimator largely recovers the filter setting, so its accuracy is an upper bound for real weather); identity caution mapping is a design assumption; pilot scale (20 pairs per condition; a formal non-inferiority claim would need about 79 pairs); reduced training schedule; only two severity extremes were driven; energy is mechanical traction work for a declared virtual vehicle, not battery energy; simulation only.
