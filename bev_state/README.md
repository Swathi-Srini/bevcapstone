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
[speed_mps, acceleration_mps2, steering, heading_error_rad,
 lane_offset_m, route_completion]
```

The BEV grid encoding is:

```text
-1.0 unknown
 0.0 free visible space
 0.5 route/lane feature, reserved for future route overlay
 0.9 ego vehicle footprint
 1.0 detected occupied object footprint
```

Object footprints are drawn from physical metre estimates scaled by the grid
resolution. The current grid covers 20 m laterally and 20 m front/back, so one
cell is 0.3125 m.

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
