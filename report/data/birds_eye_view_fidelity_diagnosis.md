# Does the BEV show the true position of a vehicle relative to the ego? (base_a replay, 2026-11-10)
Tool: `bev_rl/adaptive/diagnose_bev_fidelity.py`. True gap = distance between the ORIENTED rectangles of ego and nearest vehicle (MetaDrive pose/heading/size); BEV gap = distance between the ego block and the vehicle block as drawn in the 64x64 grid the model received (0.3125 m cells). Clear weather, density 0.05.

## What the BEV builder does (bev_state/assembler.py, unmodified teammates' code)
* `_draw_oriented_footprint(grid, x, y, width, length, value)` has NO heading parameter: every vehicle is an AXIS-ALIGNED block parallel to the ego (width across, length along), centred on its true centre, rounded to whole cells.
* Window: +-10 m lateral, +17.5 m ahead, -2.5 m behind the ego centre. `truth_objects()` keeps a vehicle only if its CENTRE is inside; outside = absent from the BEV.

## seed 1039 (oncoming vehicle drifting into the ego lane) - base_a crashes at step 318
| step | true gap m | BEV gap m | rel. heading | closing m/s | note |
|---|---|---|---|---|---|
| 308 | 14.70 | not in BEV | -115 deg | 13.9 | centre still beyond +17.5 m |
| 309 | 13.25 | 11.32 | -119 | 14.1 | first appears in the BEV: ~0.9 s before impact |
| 313 | 7.30 | 8.12 | -135 | 15.0 | |
| 316 | 3.01 | 3.64 | -148 | 15.3 | |
| 317 | 1.56 | 2.21 | -153 | 15.5 | last decision before impact: BEV shows 2.2 m, reality 1.6 m; 15.5 m/s = 1.55 m per step |

## seed 1040 (vehicle crossing from the left, roundabout merge) - base_a crashes at step 82
| step | true gap m | BEV gap m | rel. heading | note |
|---|---|---|---|---|
| 70-71 | 9.3 / 8.2 | not in BEV | -114 / -110 | |x| > 10 m: outside the lateral window |
| 72 | 7.15 | 8.03 | -105 | first appears |
| 78 | 2.67 | 4.38 | -75 | BEV overstates the gap by 1.7 m: the vehicle is nearly perpendicular, its 4.6 m length lies across the ego's path but is drawn 1.9 m wide |
| 80 | 1.09 | 2.81 | -69 | still drawn 2.8 m away, true gap 1.1 m |
| 81 | 0.49 | not in BEV | -64 | vehicle centre is 3.0 m behind the ego centre (< -2.5 m rear limit): it VANISHES from the BEV 0.1 s before impact |

## Reading
Centre positions are faithful (quantised to 0.31 m), but (1) vehicles are drawn without heading, so angled (crossing / oncoming) vehicles are mis-sized and the drawn gap can exceed the true gap by 0.5-1.7 m; (2) the window is small relative to the closing speed (13-15 m/s oncoming, ~0.9 s warning in 1039), and vehicles drop out of it by centre; (3) in my videos the BEV panel is the input at the DECISION (before the step) while the left view was rendered AFTER the step: a 0.1 s (about 1.5 m at 15 m/s) offset.
These are limits of the teammates' BEV representation, identical for every model; they are not unique to base_a. The SafeTeacher expert sees ground truth out to 45 m (capsule check), i.e. MORE than the student's window: part of the student-teacher gap is an INFORMATION gap, not only imitation error (hypothesis supported by 2 replayed seeds).

## Reproducibility finding (same day)
Same model + same seed gives different episodes depending on process history: base_a seed 1012 = 638 steps alone, 510 after seed 1040, 509 after seed 1039, and a crash at 374 steps in the video run. The initial state (map, route, ego pose, initial traffic positions) is IDENTICAL across histories, so the scenario at reset is deterministic; divergence is in traffic behaviour during the episode (RNG not reseeded at reset). Consequence: 'paired seeds' are paired on map/route/initial traffic, not on every random event; repeating the SAME evaluation order in a fresh process is reproducible, comparisons across different run orders are not exact.
