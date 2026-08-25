"""Manual MetaDrive capture for the specified front stereo + side/rear rig.

Captures five physical RGB streams: front-left/front-right (the 0.5 m stereo
pair), left, right, and rear. StereoSGBM produces a metric front depth map.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera


# Technical specification Tables 2, 3, and 5.
CAM_W, CAM_H, CAM_FOV = 1200, 900, 60
FOCAL_LENGTH_PX = 1000.0
STEREO_BASELINE_M = 0.5
CAMERA_RIGS = {
    "front_left_camera": {"pos": (-0.25, 2.0, 1.4), "hpr": (0, -5, 0)},
    "front_right_camera": {"pos": (0.25, 2.0, 1.4), "hpr": (0, -5, 0)},
    "left_camera": {"pos": (0.0, 0.0, 1.4), "hpr": (-90, -5, 0)},
    "right_camera": {"pos": (0.0, 0.0, 1.4), "hpr": (90, -5, 0)},
    "rear_camera": {"pos": (0.0, -2.0, 1.4), "hpr": (180, -5, 0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually drive and capture the spec-compliant multi-camera rig.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "captured_frames_stereo")
    return parser.parse_args()


def build_stereo_matcher() -> cv2.StereoSGBM:
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=192, blockSize=block_size,
        P1=8 * 3 * block_size ** 2, P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1, uniquenessRatio=10,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_depth(left_bgr: np.ndarray, right_bgr: np.ndarray, matcher: cv2.StereoSGBM) -> np.ndarray:
    """Specification Eq. (5): Z = f*B/d_px = 500/d_px metres."""
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0.0
    depth[valid] = FOCAL_LENGTH_PX * STEREO_BASELINE_M / disparity[valid]
    depth[(depth < 1.0) | (depth > 30.0)] = 0.0
    return depth


def build_env() -> MetaDriveEnv:
    return MetaDriveEnv({
        "use_render": True,
        "manual_control": True,
        "traffic_density": 0.1,
        "num_scenarios": 5,
        "map": 3,
        "image_observation": True,
        "norm_pixel": False,
        "stack_size": 1,
        "vehicle_config": {"image_source": "front_left_camera"},
        "sensors": {name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    })


def mount_cameras(env: MetaDriveEnv) -> None:
    for name, rig in CAMERA_RIGS.items():
        sensor = env.engine.get_sensor(name)
        sensor.track(env.agent.origin, rig["pos"], rig["hpr"])
        sensor.get_lens().setFov(CAM_FOV)


def get_frame(env: MetaDriveEnv, name: str) -> np.ndarray:
    frame = env.engine.get_sensor(name).get_rgb_array_cpu()
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def save_all_frames(env: MetaDriveEnv, step: int, output_dir: Path, matcher: cv2.StereoSGBM) -> None:
    frames = {name: get_frame(env, name) for name in CAMERA_RIGS}
    for name, frame in frames.items():
        cv2.imwrite(str(output_dir / name / f"frame_{step:05d}.png"), frame)
    depth = compute_depth(frames["front_left_camera"], frames["front_right_camera"], matcher)
    vis = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_dir / "front_depth" / f"frame_{step:05d}.png"), vis)
    np.save(output_dir / "front_depth" / f"frame_{step:05d}.npy", depth)


def main() -> int:
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")
    for name in (*CAMERA_RIGS, "front_depth"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)

    env = build_env()
    env.reset()
    mount_cameras(env)
    matcher = build_stereo_matcher()
    print("MetaDrive capture: front stereo L/R + left + right + rear.")
    print("Use Arrow Keys or WASD in the MetaDrive window. Ctrl+C stops capture.")
    try:
        for step in range(args.steps):
            _, reward, terminated, truncated, _ = env.step([0.0, 0.0])
            if step % args.save_every == 0:
                save_all_frames(env, step, args.output_dir, matcher)
                print(f"[step {step}] saved five camera frames + front SGM depth; reward={reward:.3f}")
            if terminated or truncated:
                env.reset()
                mount_cameras(env)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
