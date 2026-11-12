"""Camera-derived BEV v2 perception for the RL environment (actor-visible observation source).

Everything returned for the actor comes from camera frames (after the seeded weather perturbation) plus the
navigation route and the 6-D ego/navigation scalar, which are documented non-perception inputs. Simulator ground
truth (lane geometry, vehicle lists) is never read here.

    perception(md, info) -> (bev (7,64,64) float32, scalar (6,))
"""

from __future__ import annotations

import time

import numpy as np

from bev_state import bev_v2
from bev_state.ground_geometry import build_cell_camera_map
from bev_state.route import route_polyline_world, world_to_ego   # navigation plan, not perception
from bev_state.surface import SurfaceClassifier
from bev_state.weather import apply_weather, frame_rng


class CameraBEVPerception:
    def __init__(self, yolo_model, assembler, classifier: SurfaceClassifier | None = None, base_seed: int = 0,
                 include_route: bool = True, use_yolo: bool = True):
        self.yolo, self.asm = yolo_model, assembler
        self.clf = classifier or SurfaceClassifier.load()
        self.cmap = build_cell_camera_map()
        self.base_seed, self.include_route, self.use_yolo = int(base_seed), include_route, use_yolo
        self.kind, self.level = "clear", 0.0
        self.episode, self.step_idx = -1, 0
        self.timing = {"frames": 0.0, "weather": 0.0, "yolo": 0.0, "stereo": 0.0, "surface": 0.0, "n": 0}

    # -- weather is supplied by the environment (explicit conditioning, not detected) --
    def set_weather(self, kind: str, level: float) -> None:
        self.kind, self.level = kind, float(level)

    def on_reset(self, md) -> None:
        from manual_drive_stereo_yolo_weather import mount_camera_rig
        mount_camera_rig(md)
        self.asm._mounted_env_ids.add(id(md))
        self.episode += 1
        self.step_idx = 0

    def __call__(self, md, info):
        from manual_drive_stereo_yolo_weather import YOLO_CAMERAS, frame_from_sensor, front_stereo_depth
        from yolo.yolo_utils import run_yolo
        T = self.timing
        t0 = time.perf_counter()
        raw = {n: frame_from_sensor(md, n) for n in self.cmap.cameras}
        t1 = time.perf_counter()
        frames = {n: apply_weather(f, self.kind, self.level, n, frame_rng(self.base_seed, self.episode, self.step_idx, n))
                  for n, f in raw.items()}
        t2 = time.perf_counter()
        objects = []
        if self.use_yolo:
            dets = {n: run_yolo(self.yolo, frames[n]) for n in YOLO_CAMERAS}
            t3 = time.perf_counter()
            depth = None
            if dets.get("front_left_camera"):
                right = apply_weather(frame_from_sensor(md, "front_right_camera"), self.kind, self.level, "front_right_camera",
                                      frame_rng(self.base_seed, self.episode, self.step_idx, "front_right_camera"))
                depth = front_stereo_depth(frames["front_left_camera"], right, self.asm.matcher)
            objects = bev_v2.objects_from_detections_v2(self.asm, dets, depth)
        else:
            t3 = time.perf_counter()
        t4 = time.perf_counter()
        route_xy = None
        if self.include_route:
            route_xy = world_to_ego(route_polyline_world(md), np.asarray(md.agent.position, float), float(md.agent.heading_theta))
        bev = bev_v2.build_bev_v2(self.asm, frames, self.clf, self.cmap, objects=objects, route_ego_xy=route_xy,
                                  include_route=self.include_route)
        t5 = time.perf_counter()
        scalar = self.asm.extract_scalar_state(md, info)
        self.step_idx += 1
        for k, v in (("frames", t1 - t0), ("weather", t2 - t1), ("yolo", t3 - t2), ("stereo", t4 - t3), ("surface", t5 - t4)):
            T[k] += v
        T["n"] += 1
        return bev, scalar
