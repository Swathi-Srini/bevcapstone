# Backups and checkpoint integrity (14 November 2026)

## What is where

| Item | Tracked in git (GitHub) | Local only |
|---|---|---|
| Original checkpoints `Proposed_BEV_Driving_Model.pt` (447,729 B) and `Caution_Study_Caution_Aware_Driving_Model.pt` (456,421 B), and the other models listed in `trained_models/MODEL_MANIFEST.json` (33 entries in total), in `trained_models/` | yes, with `trained_models/MODEL_MANIFEST.json` | `bev_rl/adaptive/runs/<run>/best.pt` working copies (git-ignored, byte-identical) |
| Baselines `bc_clear_5eps_without_traffic.pt`, `bc_sparse_traffic_5eps.pt` | yes (`bev_policy/checkpoints/`, also in `trained_models/01_*`) | |
| Third-party `weights/yolov8n.pt` | no (downloadable) | yes; copy in `trained_models/06_*` is ignored |
| Camera-student checkpoints and datasets (`bev_rl/adaptive/results/camera_dagger_final/`) | only `FROZEN_MANIFEST.json` | yes (`.npz` datasets are git-ignored; checkpoints are not yet versioned) |
| Videos (`report/videos*`, 686 MB), detector-table images (257 MB), `runs/` logs, `logs/`, `report/local_only/` | no | yes |

Integrity check: all 33 entries of `trained_models/MODEL_MANIFEST.json` match their files (verified 14 Nov 2026 with `tools/cleanup/verify_checkpoints.py`).

## Local dated backup (14 November 2026)

A complete copy of the working folder was made on the maintainer's laptop (a dated folder under `D:\BEV_Capstone_Backups\`), **copied, never moved**; sources were only read.

| | |
|---|---|
| Files | 5,072 |
| Size | 1,431.8 MB (1.4 GB) |
| Included | everything in the project folder including `.git` (151 MB), results, runs, datasets, logs, videos, reports |
| Excluded | `venv/` (1.8 GB, rebuildable from `requirements.txt`) and `__pycache__/` |
| Verification | per-file SHA-256 while copying, then a second full pass over the copy: 5,072 verified, 0 failures; independently `sha256sum -c SHA256SUMS.txt`: 5,072 OK, 0 not OK |
| Files that kept changing during the copy | 0 (a training run was active; its folder was copied as a snapshot) |
| Manifest | `SHA256SUMS.txt`, `MANIFEST.json`, `BACKUP_REPORT.md` in the backup root |

Limits: the backup is on the same physical drive as the project, so it protects against deletion and overwriting mistakes, not against drive failure. It is a snapshot taken while the OV-Split training was in its third round: **repeat it when training and evaluation finish** (`tools/cleanup/backup_project.py`).

## Planned (not done yet)
* After the camera students finish: copy final `best.pt`/`iter_*.pt`, `train_round*.json`, `log.json`, `best.json` into `trained_models/07_Camera_Pipeline_Base_Models/` and append their hashes to `MODEL_MANIFEST.json` (decision pending whether `.pt` files are versioned in git).
* Re-run the dated backup.

## Re-verifying a backup

```bash
cd <backup-root>/bevcapstone && sha256sum -c ../SHA256SUMS.txt
python tools/cleanup/verify_checkpoints.py        # checks trained_models/ and runs/ against MODEL_MANIFEST.json
```
