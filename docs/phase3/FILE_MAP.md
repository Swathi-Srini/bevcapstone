# File map for phase 3

## In this branch (new files only; nothing on `main` is modified except one added pointer line in the top-level `README.md`)
| Path | What |
|---|---|
| `docs/phase3/` | these documents |
| `bev_rl/adaptive/severity_estimator.py` | image-based severity estimator (fit, eval, live filter) |
| `bev_rl/adaptive/camera_dagger_severity.py` | Model B training wrapper over `camera_dagger.py` (unchanged) |
| `bev_rl/adaptive/evaluate_severity_policies.py` | paired A / B / C evaluation with full instrumentation |
| `bev_rl/adaptive/analyze_severity_eval.py` | analysis: Wilson intervals, McNemar, bootstrap, figure |
| `bev_rl/adaptive/capture_pipeline_stills.py` | one-scene capture of depth, detections and BEV under several weathers |
| `bev_rl/adaptive/validate_teacher_identity_caution.py`, `verify_stream_equivalence.py`, `e2e_dagger_both.py` | teacher validation and checks |
| `bev_rl/adaptive/measure_retention_camera.py`, `retention_camera_summary.py`, `queue_retention_camera.sh` | earlier retention-based caution measurement (retired by the Nov 16 specification) |
| `bev_rl/adaptive/queue_severity_pipeline{,_v2,_v3}.sh` | pipeline drivers (v3 is the one that ran) |
| `bev_rl/adaptive/PREREGISTRATION_*.md`, `CHANGELOG_severity_*.md` | pre-registrations and the record of every change |
| `bev_rl/adaptive/results/severity_estimator/` | fitted estimator and its held-out evaluation |
| `bev_rl/adaptive/results/severity_eval_20seed_snapshot/` | the 120 episode files of the 20-seed evaluation, `MANIFEST_sha256.txt`, `SNAPSHOT_INFO.json` |
| `bev_rl/adaptive/results/severity_eval_20seed_analysis/` | `summary.json`, `tables.md`, `table_main.tex`, figure |
| `bev_rl/adaptive/results/teacher_identity_caution/` | teacher validation |
| `bev_rl/adaptive/results/camera_dagger_severity/` | Model B: `best.pt`, manifest, per-round training records |
| `bev_rl/adaptive/results/camera_dagger_final/e2e_ov_split_final/` | Model A: `best.pt`, `best.json`, training records |
| `bev_rl/adaptive/results/retention_camera/` | earlier retention measurements (small) |
| `report/paper_severity/` | IEEE paper draft (`main.tex`), structure-only skeleton, `references.bib`, figures (PNG and PDF), figure and self-check scripts |
| `report/Architecture_Severity_*.drawio`, `report/make_architecture_v3.py` | architecture diagrams and their generator |

## Deliberately NOT in Git
| What | Why | Where it is |
|---|---|---|
| `results/camera_dagger_severity/e2e_sevB/round*/` (DAgger data `.npz`, claim files) | `.gitignore` excludes `*.npz` and `**/claims/` | local laptop only |
| `results/severity_eval/` (original episodes) | identical to the snapshot copy; the snapshot is the stable one | local laptop only |
| `results/severity_eval_ext/` (extension, in progress) | still being written; will be added when complete | local laptop only |
| videos (`report/videos*`, about 650 MB) | `*.mp4` ignored | local laptop only |
| `logs/`, `.claude/` notes, `archive/`, `bev_rl/adaptive/runs/` | ignored by design | local laptop only |
| `report/caution_model_detector_table_images/` (about 2,000 images, 245 MB) | large, not phase 3 | local laptop only |
| `weights/yolov8n.pt` | downloaded by Ultralytics | local laptop only |
| `.aux`, `.log`, `.bbl` LaTeX byproducts, the Overleaf zip, the raw video frames used for the weather figure | regenerated or derived (the compiled `main.pdf` and the final figures are in the branch) | local laptop only |

Local-only files are not backed up anywhere else.
