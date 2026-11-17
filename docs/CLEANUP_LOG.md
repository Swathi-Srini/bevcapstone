# Cleanup log

Rule: **nothing is deleted.** Historical work is archived (moved with its history preserved), and every step is logged here. Entries are appended as steps are executed.

## 14 November 2026

| Step | Status | Notes |
|---|---|---|
| Read-only audit of the merge `7e31b6f` (216 files, 6 commits) | done | no sensitive data found; 78 files from earlier commits were included |
| Local dated backup of the project (5,072 files, 1,431.8 MB) | done | verified twice, see `docs/BACKUP_STRATEGY.md` |
| Part A: documentation and configuration (this pull request) | in review | README fixes, status index, git workflow, backup strategy, unneeded-files audit, ignore rules, read-only tools |

## Pending

| Step | Waits for | Summary |
|---|---|---|
| B1 hardcoded machine paths in 38 shell scripts and 2 helper scripts | all experiments finished | `cd /d/bev_Fresh/bevcapstone` -> `cd "$(dirname "${BASH_SOURCE[0]}")/../.."`; `PY=${PY:-./venv/Scripts/python.exe}` |
| B2 line endings: `.gitattributes`, CRLF `.sh` files | all experiments finished | must be done together with B3 because it changes file hashes |
| B3 freeze check that tolerates line endings (new manifest file; old one kept) | all experiments finished | edits `_sha`, `freeze`, `verify_frozen` in `camera_dagger.py` only |
| B4 archive the 5 superseded files to `archive/superseded_2026-11-14/` | all experiments finished | `e2e_dagger.py`, `evaluate_e2e.py`, `queue_e2e.sh`, `queue_e2e_finetune.sh`, `PREREGISTRATION_e2e.md` (and update the mention in `evaluate_n200_all_weather.py`) |
| B5 archive local partial/interrupted runs | all experiments finished | untracked; hashes recorded here before the move |
| B6 status headers in code | all experiments finished | comment-only; proven by an AST comparison |
| C1 camera-student checkpoints into `trained_models/07_*` and a second backup | training finished; needs a decision on versioning | |
| C2 smoke tests of the untested evaluation scripts | idle workers | first real run is the queued evaluation |
