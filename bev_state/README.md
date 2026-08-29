# BEV state assembly

This package owns the final state contract for the next RL/training layer. It
does not train a model.

## Output contract

`BEVStateAssembler.assemble(...)` returns:

```python
StateObservation(
    bev_grid=np.ndarray,      # shape (64, 64), float32
    scalar_state=np.ndarray,  # shape (6,), float32
    objects=tuple(...),
)
```

The scalar order is:

```text
[speed_mps, route_progress, lateral_deviation_m, heading_error_rad,
 curvature_ahead_rad_per_m, distance_to_goal_m]
```

The BEV grid encoding is:

```text
-1.0 unknown
 0.0 free visible space
 0.5 route/lane feature, reserved for future route overlay
 0.8 road-boundary feature, reserved for future boundary overlay
 0.9 ego vehicle footprint
 1.0 detected occupied object footprint
```

Object footprints are drawn from physical metre estimates scaled by the grid
resolution. The current grid covers 20 m laterally and 20 m front/back, so one
cell is 0.3125 m. The 1.9m x 4.6m ego footprint is rasterised as 6 x 15 cells
(1.875m x 4.688m), the nearest faithful grid representation.

## Minimal usage

```python
from bev_state import BEVStateAssembler

assembler = BEVStateAssembler()
observation = assembler.assemble(env=env, yolo_model=model, info=info)

bev_grid = observation.bev_grid
scalar_state = observation.scalar_state
```

For offline tests, pass detections directly:

```python
observation = assembler.assemble(
    detections_by_camera={
        "right_camera": [
            {"xmin": 550, "ymin": 500, "xmax": 650, "ymax": 700,
             "confidence": 0.9, "label": "car"}
        ]
    }
)
```

## Live integration visualiser

Run this from the repository root to inspect the actual policy input while
driving manually. It overlays the 64x64 BEV, projected object positions, and
the six scalar values. This is the required validation step before collecting
training demonstrations.

```powershell
python -m bev_state.live_visualize --weather none
python -m bev_state.live_visualize --weather fog --level 0.5 --perception-weather
```

Approach a vehicle to roughly 5-15m in front of the ego car. A valid front
stereo detection should create a red occupied footprint in the BEV. Objects
past the current 17.5m forward grid limit are intentionally not painted.

To avoid manually driving up to traffic, run the controlled system-test fixture:

```powershell
python -m bev_state.live_visualize --auto-drive --spawn-target-distance 12 --max-steps 100
```

It spawns one stationary vehicle 12m ahead in the ego lane. This is only for
validating YOLO/stereo-to-BEV projection; it must never be used as a training
input or reported as an autonomous-driving experiment.
