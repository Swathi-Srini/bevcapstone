# 06 BEV audit and BEV v2 pilot validation (2026-11-02)

## Root cause of identical BEVs
v1 `build_bev_grid` = fixed camera-FOV wedge (free) over unknown + detections + ego. No road/lane/boundary/route content
existed, and with zero traffic there were no detections -> 1 unique BEV. The 6-D scalar is simulator-derived, so weather
could not affect it either.

## Upstream geometry defects found (original module NOT edited)
* Panda3D heading H is CCW-positive: rig `left_camera` (H=-90) looks RIGHT, `right_camera` looks LEFT -> side projections mirrored.
* Rig heights are relative to the agent origin, which is 0.535 m above the road -> true camera height ~1.935 m (not 1.4).

## BEV v2 (7 ch: known, road, lane_line, boundary, obstacle, ego, route)
Camera-only perception (ground-plane cell->pixel map, NumPy MLP surface classifier, YOLO obstacles). Route = navigation plan (not perception).
GT (`ground_truth.py`) used only for calibration labels and scoring.

## Pilot validation: seed 1000, 10 frames, 5 conditions (bev_state/validation/bev_v2_pilot_seed1000.json)
See JSON. Summary: road acc on known ~94% (clear); lane-line recall ~0.6 (1-cell tolerance), solid-line precision low (0.27);
boundary F1 0.63 clear -> 0.42 fog 1.0 / 0.45 rain 1.0; known (observable) 98.8% -> 85.3% fog 1.0.
Determinism: 0 mismatches. v1 unique BEVs 1, v2 unique 10/10.
## Not validated
Obstacle projection (zero traffic), other seeds, held-out layouts, route-vs-scalar consistency (route x at y=0 averaged +0.51 m, unexplained),
BC compatibility (BC has in_channels=1), env/policy integration, weather branch, per-step cost.
