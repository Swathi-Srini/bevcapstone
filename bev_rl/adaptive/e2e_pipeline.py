"""END-TO-END environment: the bird's-eye-view grid is built from the CAMERAS, not from the simulator's vehicle list.

Every simulation step (10 Hz):
    5 MetaDrive cameras (front-left, front-right, left, right, rear)         -> teammates' rig (CAMERA_RIGS)
    GitHub weather filter on every picture (stereo pair filtered together)   -> weather.weather_utils / apply_synchronized_stereo_weather
    front StereoSGBM depth from the filtered front pair                       -> stereo_depth
    YOLOv8n on the four YOLO views                                            -> yolo.yolo_utils.run_yolo
    detections + depth -> vehicle footprints -> 64x64 grid                    -> bev_state.BEVStateAssembler.assemble()
    + the lane/route overlay (map geometry, as in the proposed model)         -> spec_bev.overlay_github_bev
Ego state (speed, progress, goal distance) and the map overlay still come from the simulator, as in the repository's target architecture ("cameras + ego/route state").
Nothing in the teammates' modules is modified; their functions are only called. The simulator's vehicle list is NOT used by the student here (the teacher/metrics may still read it).

    from bev_rl.adaptive.e2e_pipeline import E2EEnv
"""
import time

import numpy as np

from bev_rl.adaptive.caution import CautionEnv


class E2EEnv(CautionEnv):
    def __init__(self, seeds, conditions, traffic_density=0.05, yolo_conf=0.4, **kw):
        super().__init__(seeds, conditions, dropout=False, traffic_density=traffic_density, **kw)
        import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
        from yolo.yolo_utils import ensure_yolo_model
        self.yolo = ensure_yolo_model("cpu", "weights/yolov8n.pt", None, yolo_conf)
        from bev_policy.runtime import make_env
        horizon = kw.get("horizon", 800)
        self.md.close(); self.md = make_env(use_idm=False, render=False, traffic_density=traffic_density, horizon=horizon)   # same MetaDrive config + the 5-camera rig
        self.t_pipe, self.n_pipe, self.step_count = 0.0, 0, 0
        self.last_objects = []

    def reset(self, *, seed=None, options=None):
        o, i = super().reset(seed=seed, options=options)
        from manual_drive_stereo_yolo_weather import mount_camera_rig
        mount_camera_rig(self.md)                          # re-mount after every reset (as in the teammates' benchmark)
        self.step_count = 0
        self._primed = False
        return o, i

    def student_bev(self):
        from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
        from weather.weather_utils import apply_weather
        from bev_rl.adaptive.spec_bev import overlay_github_bev
        t0 = time.time()
        raw = {n: frame_from_sensor(self.md, n) for n in CAMERA_RIGS}
        kind, level = self.weather
        if kind == "clear" or level <= 0:
            frames = raw
        else:
            np.random.seed(20261114 + int(self.md.engine.global_seed) * 100003 + self.step_count)    # fresh weather draw per frame
            frames = {n: apply_weather(f, kind, level) for n, f in raw.items()}
            frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw["front_left_camera"], raw["front_right_camera"], kind, level)
        obs = self.asm.assemble(env=None, yolo_model=self.yolo, frames=frames)               # stereo depth + YOLO + footprints + grid, all teammates' code
        self.last_objects = obs.objects
        bev = overlay_github_bev(obs.bev_grid, self.md)
        self.step_count += 1; self.t_pipe += time.time() - t0; self.n_pipe += 1
        return bev

    def raw_bev(self):
        """Camera-built grid WITHOUT the overlay (for the two GitHub baselines, which were trained on the plain grid)."""
        from manual_drive_stereo_yolo_weather import CAMERA_RIGS, apply_synchronized_stereo_weather, frame_from_sensor
        from weather.weather_utils import apply_weather
        raw = {n: frame_from_sensor(self.md, n) for n in CAMERA_RIGS}
        kind, level = self.weather
        if kind == "clear" or level <= 0:
            frames = raw
        else:
            np.random.seed(20261114 + int(self.md.engine.global_seed) * 100003 + self.step_count)
            frames = {n: apply_weather(f, kind, level) for n, f in raw.items()}
            frames["front_left_camera"], frames["front_right_camera"] = apply_synchronized_stereo_weather(raw["front_left_camera"], raw["front_right_camera"], kind, level)
        self.step_count += 1
        return self.asm.assemble(env=None, yolo_model=self.yolo, frames=frames).bev_grid
