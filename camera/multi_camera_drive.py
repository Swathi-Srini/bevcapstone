"""
Person 1 - Multi-Camera MetaDrive Capture (Front, Rear, Left, Right)

Matches the architecture diagram's Sensor Layer: 4x cameras around the vehicle.
- No Administrator rights needed: uses MetaDrive's built-in manual_control
  (arrow keys / WASD captured by the native render window, not a global OS hook).
- No venv needed: runs on your regular Python install, same as before.

Run:
    python multi_camera_drive.py
"""

import os
import numpy as np
import cv2
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

CAM_W, CAM_H = 640, 400  # lower than the diagram's 1280x800 for smoother FPS; raise later if needed

# Mount offsets relative to the vehicle: pos = (x=right, y=forward, z=up) in meters
# hpr = (heading, pitch, roll) in degrees. heading 0 = facing forward, 180 = backward,
# -90 = facing left, 90 = facing right (Panda3D convention MetaDrive uses internally).
CAMERA_RIGS = {
    "front_camera": {"pos": (0.0, 2.0, 1.0), "hpr": (0, 0, 0)},
    "rear_camera":  {"pos": (0.0, -2.0, 1.0), "hpr": (180, 0, 0)},
    "left_camera":  {"pos": (-1.0, 0.0, 1.0), "hpr": (-90, 0, 0)},
    "right_camera": {"pos": (1.0, 0.0, 1.0), "hpr": (90, 0, 0)},
}


def build_env():
    config = dict(
        use_render=True,       # opens native MetaDrive window
        manual_control=True,   # arrow keys / WASD handled internally by that window - no admin needed
        traffic_density=0.1,
        num_scenarios=5,
        map=3,
        image_observation=True,
        norm_pixel=False,      # keep raw 0-255 uint8
        stack_size=1,
        vehicle_config=dict(image_source="front_camera"),  # main obs feed
        sensors={name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    )
    return MetaDriveEnv(config)


def mount_cameras(env):
    """Attach each camera sensor to the vehicle at its configured offset/orientation."""
    vehicle = env.agent
    for name, rig in CAMERA_RIGS.items():
        sensor = env.engine.get_sensor(name)
        sensor.track(vehicle.origin, rig["pos"], rig["hpr"])


def save_all_frames(env, step, out_dir):
    for name in CAMERA_RIGS:
        sensor = env.engine.get_sensor(name)
        frame = sensor.get_rgb_array_cpu()
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        fname = os.path.join(out_dir, name, f"frame_{step:05d}.png")
        cv2.imwrite(fname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def main(steps=2000, save_every=10, out_dir="./captured_frames_multi"):
    for name in CAMERA_RIGS:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    env = build_env()
    obs, info = env.reset()
    mount_cameras(env)

    print("=" * 60)
    print("Multi-Camera MetaDrive Capture: front / rear / left / right")
    print("Drive with Arrow Keys or WASD in the MetaDrive window.")
    print("Close the window (or Ctrl+C here) to stop.")
    print("=" * 60)

    try:
        for step in range(steps):
            # action is ignored by the sim while manual_control=True; keys drive it directly
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

            if step % save_every == 0:
                save_all_frames(env, step, out_dir)
                print(f"[step {step}] saved frames from all 4 cameras  reward={reward:.3f}")

            if terminated or truncated:
                print(f"Episode ended at step {step}, resetting.")
                obs, info = env.reset()
                mount_cameras(env)  # vehicle node is recreated on reset, so re-attach cameras

    except KeyboardInterrupt:
        print("\nStopped by user.")

    env.close()
    print(f"Done. Frames saved under: {out_dir}")


if __name__ == "__main__":
    main()
