# Environment record

* Interpreter: python.org CPython 3.11.5 (`C:\Users\Swathi\AppData\Local\Programs\Python\Python311`), same as the original venv. System/Store Python 3.11.9 untouched.
* venv: `D:\bev_Fresh\bevcapstone\venv` (git-ignored). Created 2026-11-01.
* Packages: exact set of the original venv recorded in `requirements-lock.txt`
  (torch 2.13.0+cpu, torchvision 0.28.0+cpu via https://download.pytorch.org/whl/cpu; numpy 2.4.6; opencv-python 5.0.0.93;
  ultralytics 8.4.115; stable-baselines3 2.9.0; gymnasium 1.3.0; metadrive-simulator 0.4.3; panda3d 1.10.13; scipy 1.17.1; ...).
  `pip` / `setuptools` / `wheel` excluded from the lock.
* Install log: `pip_install_2026-11-01.log` (failed: network ReadTimeout on a large wheel, no dependency conflict),
  `pip_install_2026-11-01_retry1.log` (succeeded, exit 0). `pip check`: "No broken requirements found."
* Side effect: MetaDrive downloaded `assets.zip` (v0.4.3) into `venv/Lib/site-packages/metadrive/` on first env creation.
* Hardware: Intel i7-1360P, 13.7 GB RAM, no CUDA GPU (CPU-only torch).

## 2026-11-01: pypdf (read-only PDF text extraction)
`pip install --target <scratchpad>/pdflib pypdf` using the venv's pip. Installed OUTSIDE the project venv and repo, only to read the Phase 3 review PDF. Not a project dependency.
