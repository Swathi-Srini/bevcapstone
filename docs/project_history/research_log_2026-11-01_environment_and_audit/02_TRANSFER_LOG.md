# Transfer log (original -> fresh)

Original `D:\meta\bevgit\bevcapstone` has NOT been modified (checked: `git status --short | wc -l` = 143 before and after).

| Date | Source | Destination | What | Verification |
|---|---|---|---|---|
| 2026-11-01 | original `bev_policy/data/clear/seed_00300..00304.npz` | scratchpad `bc_repro/clear_5eps/` (NOT in fresh repo) | 5 BC episodes (staged for BC reproduction) | sha256 of all 5 copies equal to sources |
| 2026-11-01 | (none; downloaded) | `weights/yolov8n.pt` (git-excluded via `.git/info/exclude`) | official Ultralytics YOLOv8n | sha256 `f59b3d83...` identical to original's `yolov8n.pt` |

No other file has been copied into the fresh repository. Transfers T1-T4 in `01_...` remain proposals.
Existing checkpoints used: the two committed in the fresh repo (hash-identical to the original's).
