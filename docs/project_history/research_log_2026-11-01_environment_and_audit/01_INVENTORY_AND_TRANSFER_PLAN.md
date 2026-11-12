# Inventory of the original repository and proposed transfer

Original (read-only reference): `D:\meta\bevgit\bevcapstone`
Fresh (development workspace): `D:\bev_Fresh\bevcapstone`
Recorded: 2026-11-01. Nothing in the original was modified by this audit.

## 1. Git relationship

| Item | Original | Fresh |
|---|---|---|
| HEAD | `098fa27` (Add sparse traffic BC handoff baseline) | `7c4e9ae` (merge of PR #11 containing `098fa27`) |
| Tracked-file tree vs `098fa27` | identical to `098fa27` plus 18 modified tracked files | identical to `098fa27` (`git diff 098fa27 7c4e9ae` is empty) |
| Uncommitted state | 143 status entries: 18 modified tracked files, ~120 untracked entries | clean |

**Discrepancy:** the original contains substantial unpushed *code* as well as data.
The fresh clone is a snapshot of the last pushed commit only. Unpushed code in the
original includes `bev_policy/{energy,rl_env,train_ppo,evaluate_ppo,adaptive_control,
train_visibility_bc,train_evidence_bc,calibration,reward_audit,provenance}.py`,
`bev_state/{visibility,evidence}.py` plus weather-YOLO training/benchmark scripts,
12 extra test modules, `paper/`, `research_archive/`, `tools/`, and modified versions of
18 tracked files (+1175/-100 lines). The first audit's statement "no PPO / no energy
module" is therefore true of the fresh clone only; the original has a PPO environment
and a traction-energy proxy (`energy.py`, 41 lines) that has not been reviewed yet.

## 2. Checkpoints (`bev_policy/checkpoints/`, 50 files, 41 MB in original; 2 in fresh)

Byte-identical in both: `bc_clear_5eps_without_traffic.pt` (sha256 `d66e8d39...`),
`bc_sparse_traffic_5eps.pt` (`175fd1fa...`). Both: 110,530 parameters, in_channels=1,
`BEVScalarPolicy` (Conv 32/64/96 + GAP + MLP), as committed.

Missing from fresh (48 files), grouped by architecture (parameter count, first conv):

| Group | Count | Params | Notes |
|---|---|---|---|
| BC 1-channel, same arch as committed | 6 | 110,530 | `bc_controlled_lead_v2_bev_30eps`, `bc_teacher_risk_gate_{smoke,plain_dev,weighted_dev}` |
| scalar-only MLP | 3 | 7,010 | `bc_scalar_sparse_traffic_5eps`, `bc_controlled_lead_v2_scalar_30eps`, `visibility_mixed_detector_12ep_scalar_bc_v1` |
| "visibility" spatial CNN (1-ch) | ~25 | 294,850 | different architecture from committed model (needs the modified `model.py`) |
| spatial4 / semantic variants | 2 | 294,850 / 82,178 | `bev_controlled_lead_v2_bev_spatial4_30eps`, `..semantic_spatial_30eps` |
| "evidence" 7-channel BEV | 12 | 196,946 | first conv (24,7,5,5); needs modified `model.py` |

Most of these load only with the *uncommitted* `model.py` and cannot run on fresh `HEAD`.

## 3. Datasets (`bev_policy/data/`, 192 MB, ~60 sub-directories; none in fresh)

Core BC datasets (all `bev (N,64,64) float32`, `scalar (N,6)`, `action (N,2)`):

| Directory | Episodes | Samples | Use |
|---|---:|---:|---|
| `clear` (seeds 300-324) | 25 | 11,485 | clear, no traffic; source of `bc_clear_5eps_without_traffic.pt` (seeds 300-304) |
| `sparse_traffic_5eps` | 5 | 2,448 | source of `bc_sparse_traffic_5eps.pt` |
| `sparse_traffic` | 7 | 3,219 | |
| `traffic_5eps` / `traffic_clear` / `traffic_braking` | 5 / 31 / 2 | 2,036 / 14,414 / 1,050 | |
| `controlled_lead_v1` (episodes + pilots) | 10+3+3 | 4,124 + 1,402 + 1,225 | scripted lead vehicle |
| `controlled_lead_v2` (training_30_combined, validation_n10, ...) | 30 / 10 / others | 14,042 / 4,872 / ... | |
| `teacher_risk_gate_*` | 9 each | 360-1,080 | `bev (N,1,64,64)` plus gate fields |
| `evidence_*` | 1-6 each | 4-197 | `bev (N,7,64,64)` plus `bev_weather (N,2,7,64,64)` |
| `visibility_*` (~35 dirs) | 1-30 each | 3-960 | paired clear/weather BEV, unmasked BEV, privileged `true_lead_*` fields (evaluation-only) |

Baseline vector-observation dataset: `baseline/bc_metadrive/data/train/`: 52 episodes,
29,845 samples, `observation (N,259)`, `action (N,2)`; checkpoint
`baseline/bc_metadrive/checkpoints/bc_policy_50eps.pth`; recording
`baseline/Recordings/bc_50eps_259lidarobs.mp4` (40 MB).

The `visibility_*`, `evidence_*` and `*_probe` datasets contain simulator-truth fields
(`true_lead_*`, `nearest_projection_*`). These are labelling/scoring data and must never
reach actor inputs.

## 4. Other artifacts in the original (not in fresh)

| Path | Size | Content |
|---|---|---|
| `bev_policy/outputs/` | 288 MB | evaluation JSON/CSV/logs, `rl/` PPO runs (zip checkpoints, energy pilots) |
| `bev_policy/output/`, `eval-10-eps/` | 3 MB | misc evaluation output |
| `paper/` | 86 MB | draft, figures, `active/MODEL_REGISTRY.md`, execution plans |
| `research_archive/` | 226 MB | development notes |
| `tools/` | 535 MB | embedded python311, capture scripts |
| `Panda3D-1.10/` | 112 MB | untracked Panda3D tree |
| `runs/` | 13 MB | YOLO detect runs (weather-detector training) |
| `yolov8n.pt` | 6.5 MB | official YOLOv8n weights |
| `venv/` | 1.9 GB | original environment (Python 3.11.5, see 03) |

## 5. Which pipelines can be reproduced

| Pipeline | Needs | Reproducible on fresh HEAD? |
|---|---|---|
| Clear BC (collect, train, evaluate) | committed code, `data/clear`, YOLO weights, MetaDrive | **Yes** (training from copied data; closed-loop eval needs MetaDrive + YOLO) |
| Sparse-traffic BC | same + `sparse_traffic_5eps` | **Yes** |
| Vector baseline (259-D) | committed code, baseline data | **Yes** |
| Visibility/evidence BC, PPO, scalar BC, risk gate, adaptive control | uncommitted code + data | **No** on fresh HEAD; needs the original code, not yet reviewed |

## 6. Proposed transfer (nothing copied into the fresh repository yet)

Principle: the original stays untouched; copy with hash verification; log each file in
`02_TRANSFER_LOG.md`; copy only what the next phase needs.

- **T1, minimal, for BC reproduction (about 8 MB):**
  `bev_policy/data/clear/` (25 files, 2.6 MB), `bev_policy/data/sparse_traffic_5eps/` (5, 0.6 MB),
  `yolov8n.pt` (6.5 MB; alternatively re-download and compare the hash).
- **T2, reference vector baseline (about 4 MB):** `baseline/bc_metadrive/data/train/` and
  `checkpoints/bc_policy_50eps.pth`. Skip the 40 MB recording.
- **T3, preserve the unpushed code for review, not for execution (about 1 MB):** copy the
  untracked/modified `.py`, `.md` files into `research_log/original_unpushed_code/`
  as a read-only reference snapshot (kept out of the import path), so the new work can reuse
  or deliberately discard `energy.py`, `rl_env.py`, and so on. Alternative: import them
  via a git bundle or branch; needs your decision.
- **T4, deferred until needed:** the other 48 checkpoints and the remaining datasets;
  `paper/`, `research_archive/`, `tools/`, `Panda3D-1.10/`, `outputs/`, `venv/` are **not**
  proposed for transfer.
