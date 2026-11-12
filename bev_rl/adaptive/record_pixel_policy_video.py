"""Record the IMAGE-ONLY driving policy (vcam) driving, as a front-camera video in the same format and resolution as the other report videos.

What you see: the raw 1200x900 front-left camera frame (the teammates' weather augmentation applied for fog/rain), 10 frames per second = real time, one frame per simulation step, no overlays.
What the policy sees: a SECOND camera with the same mounting and field of view, rendered at 320x240 (the size the policy was trained on), corrupted with the same weather kind and level, resized to 128x96.
So, unlike the bird's-eye-view models, THIS policy really does read camera pixels.

    python -m bev_rl.adaptive.record_pixel_policy_video --seed 5169 --weather clear:0.0 fog:1.0 rain:0.6 rain:1.0 --out report/videos
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import cv2
import numpy as np

DISPLAY_W, DISPLAY_H = 1200, 900


def make_env(seed, horizon=800, density=0.05):
    import bev_state  # noqa  (puts the YOLO+weather module on sys.path)
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from bev_rl.adaptive.degraded_env import DegradedEnv
    from bev_rl.adaptive.vanilla_camera_cnn import RENDER_H, RENDER_W

    class TwoCameraEnv(DegradedEnv):
        def __init__(self):
            super().__init__([seed], [("clear", 0.0)], mode="none", traffic_density=density, horizon=horizon, sample_seed=7)
            self.md.close()
            self.md = MetaDriveEnv({"use_render": False, "image_observation": True, "norm_pixel": False, "traffic_density": float(density), "num_scenarios": 10000,
                                    "start_seed": 10, "horizon": horizon, "random_agent_model": False, "random_lane_width": True, "random_lane_num": True,
                                    "out_of_route_done": True, "on_continuous_line_done": False, "manual_control": False, "log_level": 50,
                                    "vehicle_config": {"image_source": "front_left_camera", "show_navi_mark": False, "show_line_to_navi_mark": False, "show_navigation_arrow": False},
                                    "sensors": {"front_left_camera": (RGBCamera, RENDER_W, RENDER_H), "display_camera": (RGBCamera, DISPLAY_W, DISPLAY_H)}})

        def reset(self, *, seed=None, options=None):
            o, i = super().reset(seed=seed, options=options)
            from manual_drive_stereo_yolo_weather import CAMERA_RIGS, CAM_FOV
            pos, hpr = CAMERA_RIGS["front_left_camera"]
            for name in ("front_left_camera", "display_camera"):      # same rig, remounted after EVERY reset
                s = self.md.engine.get_sensor(name)
                s.track(self.md.agent.origin, pos, hpr)
                s.get_lens().setFov(CAM_FOV)
            return o, i
    return TwoCameraEnv()


def record(seed, weather, out, env, net, torch):
    from bev_rl.adaptive.vanilla_camera_cnn import act_net, corrupt, to_net_input
    from manual_drive_stereo_yolo_weather import frame_from_sensor
    from weather.weather_utils import apply_weather
    env.reset(options={"scenario_seed": seed, "weather": ("clear", 0.0)})
    crng = np.random.default_rng(seed * 13 + 5)              # same corruption stream as the 200-episode evaluation
    vw, t, info, speeds, t0 = None, 0, None, [], time.time()
    while True:
        policy_frame = corrupt(frame_from_sensor(env.md, "front_left_camera"), weather[0], weather[1], crng)
        act = act_net(net, torch, to_net_input(policy_frame))
        _, _, te, tr, info = env.step(act)
        t += 1
        speeds.append(info["mean_speed"])
        frame = frame_from_sensor(env.md, "display_camera")
        if weather[0] != "clear":
            np.random.seed(20261110 + seed * 1000 + t)
            frame = apply_weather(frame, weather[0], weather[1])
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        if te or tr:
            break
    vw.release()
    return {"arm": "vcam", "seed": seed, "weather": f"{weather[0]}:{weather[1]}", "outcome": info["outcome"], "steps": t,
            "mean_speed": float(np.mean(speeds)), "frame_size": f"{frame.shape[1]}x{frame.shape[0]}", "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5169)
    ap.add_argument("--weather", nargs="+", default=["clear:0.0", "fog:1.0", "rain:0.6", "rain:1.0"])
    ap.add_argument("--out", default="report/videos")
    ap.add_argument("--run", default="vcam")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch, ultralytics  # noqa
    from bev_rl.adaptive.vanilla_camera_cnn import RUNS, build_net
    ck = torch.load(RUNS / a.run / "best.pt", map_location="cpu", weights_only=True)
    net = build_net()
    net.load_state_dict(ck["model_state"])
    net.eval()
    env = make_env(a.seed)
    for w in a.weather:
        kind, _, level = w.partition(":")
        weather = (kind, float(level or 0))
        tmp = out_dir / f"_tmp_pixel_policy_{kind}{level}.mp4"
        rec = record(a.seed, weather, tmp, env, net, torch)
        wname = "clear" if kind == "clear" else f"{kind}_{float(level):g}"
        final = out_dir / f"front_camera__seed{a.seed}__pixel_policy_front_image_only__{wname}__{rec['outcome']}.mp4"
        tmp.replace(final)
        rec["file"] = final.name
        print(json.dumps(rec), flush=True)
        with open(out_dir / "pixel_policy_index.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    env.close()


if __name__ == "__main__":
    main()
