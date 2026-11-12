# Known limitations / threats to validity
1. BEV validation so far: 1 seed, 10 frames, intensities 0.45 and 1.0 only.
2. Lane-line quality weak: solid-line precision 0.27 (clear), boundary recall 0.51; merged-class labels are noisy.
3. Obstacle projection validated only in a 12-placement clear-weather test (1.6 m mean error; 10/12 detected; front stereo uses an orientation prior); no weather, moving traffic or multi-class test.
4. Route channel is the navigation plan and the 6-D scalar is simulator-derived; neither is camera-derived. Route offset was checked: it matches the scalar lateral offset (corr -0.995, opposite sign); the earlier +0.51 m was real BC drift.
5. Fog and rain are post-hoc image perturbations of a clean render, not physically simulated optics; rain streaks are not physically calibrated.
6. Flat-ground assumption for the cell->pixel mapping and fog depth.
7. Classifier trained on 4 calibration seeds, PG maps, zero traffic.
8. Cameras cover about 53% of the grid; the rear is blind.
9. Energy model is an approximation (bev_rl/energy.py).
10. BC checkpoint (1-channel BEV) incompatible with BEV v2.
11. Frames are BGR throughout; fog airlight tint is channel-swapped relative to its RGB docstring (cosmetic, consistent across train/eval). 12. RL uses native 600x450 cameras; YOLO/stereo not used in zero-traffic training (obstacle channel empty); obstacle path validated separately.
