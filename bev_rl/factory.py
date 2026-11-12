"""Construct the perception->BEV pipeline and environments (import order matters on Windows)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# torch / ultralytics must be imported before MetaDrive/Panda3D (DLL conflict noted in the repo).
import torch  # noqa: F401
import ultralytics  # noqa: F401

from bev_rl.env import BEVDrivingEnv
from bev_rl.reward import RewardConfig


class FullPerception:
    """The unmodified repository pipeline: 5 cameras, YOLO on 4 views, front SGBM, BEV grid."""

    def __init__(self, yolo_model, assembler):
        self.yolo, self.assembler = yolo_model, assembler

    def on_reset(self, md) -> None:
        from manual_drive_stereo_yolo_weather import mount_camera_rig
        mount_camera_rig(md)                       # idempotent; re-attach cameras to the (re)spawned agent
        self.assembler._mounted_env_ids.add(id(md))

    def __call__(self, md, info):
        obs = self.assembler.assemble(env=md, yolo_model=self.yolo, info=info)
        return obs.bev_grid, obs.scalar_state


class LazyStereoPerception(FullPerception):
    """Same output as ``FullPerception`` (proven bit-identical in research_log/04), ~2x faster.

    ``BEVStateAssembler.objects_from_detections`` reads the stereo depth map only for
    detections from the front camera; side/rear detections use monocular ground-plane
    depth. So when YOLO finds nothing in the front view, the 1.1 s SGBM computation cannot
    influence the BEV and is skipped. YOLO still runs on all four views every step.
    """

    def __call__(self, md, info):
        import numpy as np
        from manual_drive_stereo_yolo_weather import CAM_H, CAM_W, YOLO_CAMERAS, frame_from_sensor, front_stereo_depth
        from yolo.yolo_utils import run_yolo
        asm = self.assembler
        frames = {name: frame_from_sensor(md, name) for name in YOLO_CAMERAS}
        detections = {name: run_yolo(self.yolo, frames[name]) for name in YOLO_CAMERAS}
        depth = None
        if detections.get("front_left_camera"):
            right = frame_from_sensor(md, "front_right_camera")
            depth = front_stereo_depth(frames["front_left_camera"], right, asm.matcher)
        objects = asm.objects_from_detections(detections, depth)
        return asm.build_bev_grid(objects), asm.extract_scalar_state(md, info)


class CachedBEVPerception(LazyStereoPerception):
    """Fast surrogate for CLEAR + ZERO-TRAFFIC scenes where the BEV is a constant.

    The scalar state is always computed live. The BEV is the constant empty-scene grid
    (``build_bev_grid([])``) EXCEPT on audited frames (the first frame of every episode and
    every ``audit_every``-th frame afterwards), where the real YOLO/BEV pipeline runs and its
    output is what the policy receives. Every audited frame is compared with the cached grid;
    a nonzero ``audit_mismatches`` means the surrogate premise is false for that run.
    """

    def __init__(self, yolo_model, assembler, audit_every: int = 25):
        super().__init__(yolo_model, assembler)
        self.audit_every = int(audit_every)
        self.reference = assembler.build_bev_grid([])
        self.calls = self.audits = self.audit_mismatches = 0
        self._since_audit = 0

    def on_reset(self, md) -> None:
        super().on_reset(md)
        self._since_audit = 0

    def __call__(self, md, info):
        self.calls += 1
        if self._since_audit % self.audit_every == 0:
            self._since_audit += 1
            bev, scalar = super().__call__(md, info)
            self.audits += 1
            if not np.array_equal(bev, self.reference):
                self.audit_mismatches += 1
            return bev, scalar
        self._since_audit += 1
        return self.reference.copy(), self.assembler.extract_scalar_state(md, info)


def load_yolo(weights: str = "weights/yolov8n.pt", device: str = "cpu"):
    from bev_state import BEVStateAssembler  # noqa: F401  (adds YOLO+weather module to sys.path)
    from yolo.yolo_utils import ensure_yolo_model
    return ensure_yolo_model(device, weights, None, 0.4)     # same confidence as the BC pipeline


def make_driving_env(scenario_seeds: Sequence[int], scalar_mean, scalar_std, *, horizon: int = 800,
                     sample_seed: int = 0, sequential: bool = False, reward_cfg: RewardConfig | None = None,
                     yolo_weights: str = "weights/yolov8n.pt", device: str = "cpu", perception=None,
                     lazy_stereo: bool = True, cached_bev: bool = False, audit_every: int = 25):
    from bev_policy.runtime import make_env as make_md_env
    from bev_state import BEVStateAssembler
    if perception is None:
        yolo, asm = load_yolo(yolo_weights, device), BEVStateAssembler()
        if cached_bev:
            perception = CachedBEVPerception(yolo, asm, audit_every)
        else:
            perception = (LazyStereoPerception if lazy_stereo else FullPerception)(yolo, asm)
    # Identical MetaDrive configuration to BC collection/evaluation, with ZERO traffic.
    md = make_md_env(use_idm=False, render=False, traffic_density=0.0, horizon=horizon)
    return BEVDrivingEnv(scenario_seeds, scalar_mean, scalar_std, perception, reward_cfg, horizon,
                         sample_seed, sequential, md)


def make_v2_env(scenario_seeds: Sequence[int], scalar_mean, scalar_std, *, weather_conditions=(("clear", 0.0),),
                weather_input: str = "explicit", horizon: int = 800, sample_seed: int = 0, sequential: bool = False,
                reward_cfg: RewardConfig | None = None, cam_scale: float = 0.5, include_route: bool = True,
                use_yolo: bool = False, traffic_density: float = 0.0, yolo_weights: str = "weights/yolov8n.pt"):
    """Camera-BEV-v2 environment. Weather is applied to the camera frames (seeded) and supplied explicitly (or zeroed)."""
    from bev_policy.runtime import make_env as make_md_env
    from bev_rl.perception_v2 import CameraBEVPerception
    from bev_state import BEVStateAssembler
    from bev_state.bev_v2 import N_CHANNELS
    asm = BEVStateAssembler()
    perception = CameraBEVPerception(load_yolo(yolo_weights) if use_yolo else None, asm, base_seed=sample_seed,
                                     include_route=include_route, use_yolo=use_yolo)
    md = make_md_env(use_idm=False, render=False, traffic_density=traffic_density, horizon=horizon, cam_scale=cam_scale)
    return BEVDrivingEnv(scenario_seeds, scalar_mean, scalar_std, perception, reward_cfg, horizon, sample_seed, sequential, md,
                         bev_channels=N_CHANNELS, weather_conditions=weather_conditions, weather_input=weather_input)
