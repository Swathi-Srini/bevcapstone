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
|- requirements.txt          # Shared Python dependencies, including MetaDrive 0.4.3
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

The camera prototype uses five physical streams: a front-left/front-right stereo pair plus left, right, and rear cameras. The multi-camera capture utility and the front StereoSGBM depth prototype are implemented. BEV construction, the scalar-state extractor, CNN/MLP policy, and PPO training are **not implemented yet**.

### Rules for future implementations

- Use MetaDrive's two-value action contract: `[steering, throttle/brake]`.
- Do not feed exact simulator NPC positions/sizes to the learned policy. They are debugging/evaluation ground truth only.
- Route/lane geometry and ego localisation are allowed structured inputs.
- `free` and `unknown` must be distinct BEV states; visibility-aware BEV needs an explicit unknown/knownness representation.
- The YOLO/weather runner defaults to display-only fog/rain so that synthetic, unmatched weather streaks do not corrupt stereo correspondence. Use `--perception-weather` to run YOLO and StereoSGBM on the weathered frames and explicitly test weather-degraded perception.
- On Windows CPU laptops, favour headless/small-camera debugging. MetaDrive onscreen rendering can exhaust shared graphics memory.

## Environments and dependencies

Python 3.11 is recommended. The repository now has one shared root dependency file, pinned to `metadrive-simulator==0.4.3`. Use one root-level environment for both the preserved baseline and the YOLO/weather prototype.

### Shared environment

```powershell
cd <repository-root>
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

On Windows CPU machines, install the official CPU PyTorch wheels after the shared requirements. This prevents GPU/DLL loading issues on machines without a compatible CUDA setup:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
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

Run from the repository root. The script resolves its local `yolo` and `weather` packages itself.

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather none
```

Controls: `W/A/S/D` drive; `Q` quits.

```powershell
# Display weather while retaining clean stereo correspondence (default).
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather rain --level 0.5 --traffic-density 0.5

# Weather-degraded perception experiment: YOLO and StereoSGBM receive weathered frames.
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather fog --level 0.5 --perception-weather

# Finite automated smoke run; output is written to integrated_output\detection_log.csv.
.\venv\Scripts\python.exe ".\YOLO+weather module\manual_drive_stereo_yolo_weather.py" --device cpu --weather none --max-steps 100
```

Verify the reusable stereo module before a handoff:

```powershell
Push-Location '.\YOLO+weather module'
..\venv\Scripts\python.exe .\stereo_depth\test_module.py
Pop-Location
```

### YOLO-weather unit and integration tests

The non-GUI suite checks camera calibration, SGBM configuration and depth conversion, stereo uncertainty, four-camera coordinate projection, weather augmentation, YOLO class/confidence filtering, visualization helpers, and the saved-image weather benchmark with a mocked model. It does not start MetaDrive or require `yolov8n.pt`.

```powershell
.\venv\Scripts\python.exe -m unittest discover -s ".\YOLO+weather module\tests" -v
```

The first run downloads the official `yolov8n.pt` weights when network access is available; alternatively, provide a local weight file with `--yolo-model <path>`. Use `--device cpu` unless a compatible CUDA setup is verified.

### Saved-image weather benchmark

This benchmark runs the same YOLO model on a clean image and a synthetic fog/rain version, then writes both annotated images and `summary.csv` for a direct detection-count/confidence comparison:

```powershell
.\venv\Scripts\python.exe ".\YOLO+weather module\benchmark_weather_perception.py" --image .\path\to\image.png --weather fog --level 0.5 --device cpu
```

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
