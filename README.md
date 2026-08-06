# BEV Autonomous Driving Research Project

This repository contains a MetaDrive-based autonomous-driving research project. The target system is a vision-derived Bird's-Eye View (BEV) controller trained with behavioural cloning (BC) and later fine-tuned with PPO.

**The final simulator is MetaDrive, not CARLA.** Earlier CARLA work is retained as exploratory/design context; it is not the implementation target.

## Repository layout

```text
bevcapstone/
|- baseline/
|  |- bc_metadrive/          # Preserved 259-D vector-observation BC baseline
|  `- Recordings/            # Local baseline rollout evidence (ignored by Git)
|- YOLO+weather module/      # Standalone MetaDrive camera + YOLO prototype
|- venv/                     # Local Python environment (ignored by Git)
|- requirements.txt
`- .gitignore
```

Do not overwrite `baseline/bc_metadrive/` during BEV development. It is the comparison experiment. Future BEV code should live separately, for example in a top-level `bev_policy/` folder.

## Target architecture

```text
MetaDrive cameras + ego/route state
  -> YOLO detections + depth/projection + lane/route geometry
  -> 64 x 64 BEV grid + 6-D scalar state
  -> CNN branch (BEV) + MLP branch (scalars)
  -> feature fusion + two-output policy
  -> MetaDrive action [steering, throttle/brake]
  -> MetaDrive environment, reward, and next observation
```

The target camera concept uses five physical streams: a front-left/front-right stereo pair plus left, right, and rear cameras. Multi-camera capture, stereo depth, BEV construction, the scalar-state extractor, CNN/MLP policy, and PPO training are **not implemented yet**.

### Rules for future implementations

- Use MetaDrive's two-value action contract: `[steering, throttle/brake]`.
- Do not feed exact simulator NPC positions/sizes to the learned policy. They are debugging/evaluation ground truth only.
- Route/lane geometry and ego localisation are allowed structured inputs.
- `free` and `unknown` must be distinct BEV states; visibility-aware BEV needs an explicit unknown/knownness representation.
- The YOLO/weather module currently detects on clean RGB and adds fog/rain only for display. It does not yet demonstrate weather-degraded perception.
- On Windows CPU laptops, favour headless/small-camera debugging. MetaDrive onscreen rendering can exhaust shared graphics memory.

## Environments and dependencies

Python 3.11 is recommended. The modules currently have incompatible MetaDrive declarations:

- `baseline/bc_metadrive/requirements.txt` requests the latest GitHub MetaDrive source.
- `YOLO+weather module/requirements.txt` pins `metadrive-simulator==0.4.3`.

Do not blindly install both requirement files into one environment. Until a final version is pinned, use separate environments when independently reproducing the modules.

### Baseline environment

```powershell
py -3.11 -m venv .venv-baseline
.\.venv-baseline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\baseline\bc_metadrive\requirements.txt
```

### YOLO/weather environment

```powershell
py -3.11 -m venv .venv-yolo
.\.venv-yolo\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r '.\YOLO+weather module\requirements.txt'
```

If activation is blocked, run this once per PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Run the baseline independently

Run all commands from the baseline directory because data/checkpoint paths are relative.

```powershell
cd .\baseline\bc_metadrive
```

Recommended order:

```powershell
# Diagnostics; do not start the full driving renderer
python inspect_dataset.py
python inspect_actions.py
python test_dataset.py
python test_model.py

# Manual MetaDrive collection; creates data/train/episode_*.npz
python collect_demos.py

# BC training; creates checkpoints/bc_policy.pth
python train_bc.py
```

Optional checks:

```powershell
python test_action_timing.py
python test_training_sample.py
```

`evaluate_bc.py` intentionally loads `checkpoints/bc_policy_50eps.pth`, the frozen comparison model rather than the most recently generated checkpoint.

```powershell
python evaluate_bc.py
```

`test.py`, `collect_demos.py`, and `evaluate_bc.py` use onscreen MetaDrive rendering and may fail with out-of-memory errors on a low-memory Windows laptop. Close other heavy applications and run one MetaDrive process at a time.

## Run YOLO + weather independently

Run from inside the module directory so the local `yolo` and `weather` packages resolve.

```powershell
cd '.\YOLO+weather module'
python .\manual_drive_visualize.py --device cpu --weather none
```

Controls: `W/A/S/D` drive; `Q` quits.

```powershell
python .\manual_drive_visualize.py --device cpu --weather fog --level 0.5
python .\manual_drive_visualize.py --device cpu --weather rain --level 0.7
python .\manual_drive_visualize.py --device cpu --weather all --level 0.5
python .\manual_drive_visualize.py --device cpu --weather none --max-steps 100
```

The first run may download `yolov8n.pt`. Use `--device cpu` unless a compatible CUDA setup is verified.

## Experiment contracts

| Experiment | Observation | Policy | Action |
|---|---|---|---|
| Preserved baseline | 259-D MetaDrive vector | MLP | `[steering, throttle/brake]` |
| Proposed BEV policy | 64 x 64 BEV + 6-D scalar state | CNN + MLP fusion | `[steering, throttle/brake]` |

Proposed curriculum:

```text
BEV behavioural cloning
  -> PPO without traffic
  -> energy-aware reward
  -> controlled reduced visibility
  -> traffic and scenario randomisation
```

Evaluate using matched MetaDrive scenarios: success, collision count, progress, energy per metre, speed, lane deviation, heading error, and jerk. Do not directly compare CARLA and MetaDrive metrics.

## Generated artifacts and Git

Git ignores local environments, caches, demo data (`*.npz`), weights/checkpoints (`*.pt`, `*.pth`, `*.onnx`), videos, local configuration (`.env`), YOLO output, and standard experiment-run directories. Store artifacts externally or release them deliberately when exact reproduction is needed.

Ignore rules do not remove a file that was already committed. Before pushing a large artifact, verify it with:

```powershell
git check-ignore -v <path>
git ls-files -- <path>
```

The first command shows the ignore rule; the second shows whether the path is already tracked.
