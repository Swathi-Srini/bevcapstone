# Architecture (state on 2026-11-02)

```
5 cameras 1200x900 (front stereo pair, left, right, rear)
  -> weather perturbation (seeded; depth-dependent fog, alpha-composited rain streaks)
  -> ground-plane cell->pixel map (64x64 cells, 0.3125 m) -> 15 patch features -> NumPy MLP surface classifier
  -> YOLO + stereo depth (front) / ground ray (side, rear) -> obstacle footprints
  => BEV v2, (7,64,64) float32: known, road, lane_line, boundary, obstacle, ego, route

BEV v2 -> CNN encoder ----------------------------+
6-D ego state [speed, route fraction, lateral offset,|
  heading error, curvature ahead, distance to goal] -> MLP --+--> fusion --> PPO actor (steer, throttle/brake) + critic
weather one-hot {clear,fog,rain} + intensity (4-D, explicit input) -> small MLP --+
```

Grid: ego frame, x right, y forward; +-10 m lateral, +17.5 m ahead, -2.5 m behind. Unknown is explicit: known=0 zeroes all perception channels.

Not yet integrated: BEV v2 in the `bev_rl` env/policy (the policy still expects the 1-channel v1 BEV) and the weather branch.

Non-perception inputs (documented): the 6-D scalar is simulator-derived; the `route` channel is the navigation plan. Ground truth (`bev_state/ground_truth.py`) is for labels, validation and evaluation only and must not be imported by the actor path.
