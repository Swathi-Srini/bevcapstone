"""Spec-compliant BEV overlay (docs/AV_Carla_BEV.pdf, Sec. 5): HD-map route centreline and road boundaries.

Policy-side module (does not modify bev_state). Values follow spec Table 7 exactly:
    0.0 free, 0.5 route centreline (HD map), 0.8 road boundary / lane edges (HD map), 1.0 ego footprint / occupied.
Grid: 64x64, lateral +-10 m, +17.5 m ahead, -2.5 m behind, ego at row 56 / col 32 (Table 6).
Obstacles are NOT drawn here: in the spec they come from YOLO + stereo (zero-traffic scenes have none).
Route and boundaries come from the simulator map (the spec's 'HD map', Table 1) and are expressed in the ego frame.
"""
from __future__ import annotations

import numpy as np

GRID, LAT, FWD, REAR = 64, 10.0, 17.5, 2.5
EXT = FWD + REAR
ROUTE, BOUND, OCC = 0.5, 0.8, 1.0
POSE_ERROR = None   # (dx_right_m, dy_fwd_m, dtheta_rad): localisation error of the ego pose against the HD map (None = perfect). Set per episode by evaluation code.


def _to_cells(xr: np.ndarray, zf: np.ndarray):
    col = ((xr + LAT) / (2 * LAT) * GRID).astype(int)
    row = ((1.0 - (zf + REAR) / EXT) * GRID).astype(int)
    ok = (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID)
    return row[ok], col[ok]


def _lane_chain(nav, max_hops: int = 3):
    lanes = [nav.current_lane]
    ref = list(nav.next_ref_lanes or [])
    for _ in range(max_hops):
        if not ref:
            break
        idx = min(getattr(lanes[-1], "index", (0, 0, 0))[-1] if hasattr(lanes[-1], "index") else 0, len(ref) - 1)
        lanes.append(ref[idx])
        break          # one lookahead lane is enough for 17.5 m at the lane lengths used here
    return lanes


def map_layers(md):
    """HD-map layers in the ego-frame BEV grid: (route_centreline_mask, lane_edge_mask), both (64,64) bool. No camera involved."""
    agent = md.agent
    pos = np.asarray(agent.position, float)
    th = float(agent.heading_theta)
    fwd = np.array([np.cos(th), np.sin(th)])
    right = np.array([np.sin(th), -np.cos(th)])
    if POSE_ERROR is not None:      # map is drawn relative to the ESTIMATED pose (true pose + error)
        pos = pos + POSE_ERROR[0] * right + POSE_ERROR[1] * fwd; th = th + POSE_ERROR[2]
        fwd = np.array([np.cos(th), np.sin(th)]); right = np.array([np.sin(th), -np.cos(th)])
    route = np.zeros((GRID, GRID), bool); edge = np.zeros((GRID, GRID), bool)
    nav = agent.navigation
    lanes = _lane_chain(nav)
    s0, _ = lanes[0].local_coordinates(agent.position)
    for k, lane in enumerate(lanes):
        s_start = max(0.0, s0 - 4.0) if k == 0 else 0.0
        s = np.arange(s_start, lane.length, 0.15)
        if s.size == 0:
            continue
        w = np.array([lane.width_at(float(x)) if hasattr(lane, "width_at") else lane.width for x in s])
        for lat_scale, target in ((0.0, route), (0.5, edge), (-0.5, edge)):
            pts = np.array([lane.position(float(si), float(wi * lat_scale)) for si, wi in zip(s, w)])
            d = pts - pos
            r, c = _to_cells(d @ right, d @ fwd)
            target[r, c] = True
    return route, edge


def overlay_github_bev(grid: np.ndarray, md) -> np.ndarray:
    """Fill the GitHub encoding's RESERVED values (route 0.5, road boundary 0.8) from the HD map. The GitHub grid builder is
    not modified: this wraps its output. Ego (0.9) and detected vehicles (1.0) are never overwritten; map layers also fill
    camera-unknown (-1) cells because they come from the map, not the cameras."""
    route, edge = map_layers(md)
    out = grid.copy()
    free = (out == 0.0) | (out == -1.0)
    out[edge & free] = BOUND
    out[route & free] = ROUTE
    return out


def build_spec_bev(md) -> np.ndarray:
    agent = md.agent
    pos = np.asarray(agent.position, float)
    th = float(agent.heading_theta)
    fwd = np.array([np.cos(th), np.sin(th)])
    right = np.array([np.sin(th), -np.cos(th)])
    grid = np.zeros((GRID, GRID), np.float32)
    nav = agent.navigation
    lanes = _lane_chain(nav)
    cur = lanes[0]
    s0, _ = cur.local_coordinates(agent.position)
    for k, lane in enumerate(lanes):
        s_start = max(0.0, s0 - 4.0) if k == 0 else 0.0
        s = np.arange(s_start, lane.length, 0.15)
        if s.size == 0:
            continue
        w = np.array([lane.width_at(float(x)) if hasattr(lane, "width_at") else lane.width for x in s])
        for lat_scale, val in ((0.0, ROUTE), (0.5, BOUND), (-0.5, BOUND)):
            pts = np.array([lane.position(float(si), float(wi * lat_scale)) for si, wi in zip(s, w)])
            d = pts - pos
            r, c = _to_cells(d @ right, d @ fwd)
            if val == ROUTE:
                grid[r, c] = np.maximum(grid[r, c], ROUTE) if False else ROUTE
            else:
                grid[r, c] = BOUND
    # ego footprint 2.0 x 4.5 m, centred at the ego position (value 1.0)
    ex, ez = np.meshgrid(np.arange(-1.0, 1.0 + 1e-6, 0.15), np.arange(-2.25, 2.25 + 1e-6, 0.15))
    r, c = _to_cells(ex.ravel(), ez.ravel())
    grid[r, c] = OCC
    return grid


def ascii_view(grid: np.ndarray, step: int = 2) -> str:
    ch = {0.0: ".", ROUTE: "-", BOUND: "|", OCC: "E"}
    return "\n".join("".join(ch[round(float(v), 1)] for v in row[::step]) for row in grid[::step])
