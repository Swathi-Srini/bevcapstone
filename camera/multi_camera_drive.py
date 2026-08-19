"""
<<<<<<< HEAD
Person 1 - Multi-Camera MetaDrive Capture (Front, Rear, Left, Right)

Matches the architecture diagram's Sensor Layer: 4x cameras around the vehicle.
- No Administrator rights needed: uses MetaDrive's built-in manual_control
  (arrow keys / WASD captured by the native render window, not a global OS hook).
- No venv needed: runs on your regular Python install, same as before.

Run:
    python multi_camera_drive.py
=======
Person 1 - MetaDrive Multi-Camera Capture (Front-Stereo, Left, Right, Rear)

This is the FULL spec version matching Table 2/3/5 of the internal spec PDF
("Vision-Based BEV Perception for Energy-Optimal AV using PPO"):
  - 4 directions: front, left, right, rear
  - Front is a STEREO PAIR (two cameras, 0.5m baseline) -> 5 sensors total
  - Stereo depth computed with the exact OpenCV StereoSGBM params from Table 5

If you only need front/left/right (no stereo, no rear) for quick MetaDrive
exploration, use the earlier `multi_camera_drive.py` instead -- that was the
throwaway "let's learn the tool" version. THIS file is the one that matches
your actual capstone deliverable.

Run:
    python metadrive_stereo_capture.py
>>>>>>> camera
"""

import os
import numpy as np
import cv2
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera

<<<<<<< HEAD
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
=======
# --- Camera resolution / FOV (spec PDF, Table 3) ---------------------------
# Doc spec is 1200x900 @ 60deg FOV. That's heavy for a live-driving loop, so
# CAM_W/CAM_H default lower for FPS -- bump to (1200, 900) for a final-quality
# capture pass once driving is dialed in.
CAM_W, CAM_H = 640, 480          # doc spec: 1200 x 900
CAM_FOV = 60                     # degrees, horizontal (Table 3)
FOCAL_LENGTH_PX = 1000           # derived in doc Eq.(1): f = (W/2)/tan(FOV/2), at 1200px width
STEREO_BASELINE_M = 0.5          # Table 2/Sec 2.3: 0.5m between the two front cameras

# NOTE: FOCAL_LENGTH_PX above is the doc's value computed for a 1200px-wide
# image. If you run at a different CAM_W, MetaDrive's actual focal length in
# pixels will differ (f scales with resolution for a fixed FOV) -- recompute
# via f = (CAM_W/2) / tan(radians(CAM_FOV/2)) if you change CAM_W and need
# metric-accurate depth. At the doc's own resolution (1200) this constant is
# already correct.

# --- Mount rig, converted from the doc's (x=fwd, z=height, yaw, pitch) into
# MetaDrive's (x=right, y=forward, z=up) / (heading, pitch, roll) convention.
# Doc Table 2 (per-direction):
#   Front  x=+2.0  z=1.4  yaw=0    pitch=-5
#   Left   x= 0.0  z=1.4  yaw=-90  pitch=-5
#   Right  x= 0.0  z=1.4  yaw=+90  pitch=-5
#   Rear   x=-2.0  z=1.4  yaw=180  pitch=-5
# Doc's "x" column is the vehicle's FORWARD offset, not lateral -- left/right
# cameras sit on the centerline and just rotate to face sideways.
# The front stereo pair (Sec 2.3) is mounted symmetrically 0.25m either side
# of centerline at the front, i.e. lateral offset = +/- STEREO_BASELINE_M/2.
HALF_BASELINE = STEREO_BASELINE_M / 2.0  # 0.25 m

CAMERA_RIGS = {
    # Front stereo pair. In MetaDrive's (x=right, y=forward) convention,
    # "left" of centerline is -x, "right" of centerline is +x.
    "front_left_camera":  {"pos": (-HALF_BASELINE, 2.0, 1.4), "hpr": (0, -5, 0)},
    "front_right_camera": {"pos": (+HALF_BASELINE, 2.0, 1.4), "hpr": (0, -5, 0)},

    "left_camera":  {"pos": (0.0, 0.0, 1.4), "hpr": (-90, -5, 0)},
    "right_camera": {"pos": (0.0, 0.0, 1.4), "hpr": (90, -5, 0)},
    "rear_camera":  {"pos": (0.0, -2.0, 1.4), "hpr": (180, -5, 0)},
}

# Which single camera feeds env.step()'s main `obs` (MetaDrive requires one).
PRIMARY_CAMERA = "front_left_camera"


# --- Stereo matcher, exact params from spec PDF Table 5 ---------------------
def build_stereo_matcher():
    """OpenCV StereoSGBM configured exactly per Table 5 of the spec PDF."""
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=192,          # Table 5 / Eq.(7): covers ~3m-30m, must be multiple of 16
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,  # = 600
        P2=32 * 3 * block_size ** 2, # = 2400
        disp12MaxDiff=1,
        uniquenessRatio=10,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_depth(left_bgr, right_bgr, matcher):
    """
    Stereo depth per spec PDF Sec 3.1-3.2.
        Z = f * B / d_px   (Eq. 5, with f*B = 1000 * 0.5 = 500)
    Returns a float32 depth map in meters, 0 where disparity is invalid.
    """
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0  # SGBM returns fixed-point x16

    depth = np.zeros_like(disparity)
    valid = disparity > 0.0
    depth[valid] = (FOCAL_LENGTH_PX * STEREO_BASELINE_M) / disparity[valid]
    # Clip to the doc's stated operating range (Table 5: 1m - 30m)
    depth[(depth < 1.0) | (depth > 30.0)] = 0.0
    return depth


def build_env():
    config = dict(
        use_render=True,
        manual_control=True,   # arrow keys / WASD, native MetaDrive window
>>>>>>> camera
        traffic_density=0.1,
        num_scenarios=5,
        map=3,
        image_observation=True,
<<<<<<< HEAD
        norm_pixel=False,      # keep raw 0-255 uint8
        stack_size=1,
        vehicle_config=dict(image_source="front_camera"),  # main obs feed
=======
        norm_pixel=False,
        stack_size=1,
        vehicle_config=dict(image_source=PRIMARY_CAMERA),
>>>>>>> camera
        sensors={name: (RGBCamera, CAM_W, CAM_H) for name in CAMERA_RIGS},
    )
    return MetaDriveEnv(config)


def mount_cameras(env):
    """Attach each camera sensor to the vehicle at its configured offset/orientation."""
    vehicle = env.agent
    for name, rig in CAMERA_RIGS.items():
        sensor = env.engine.get_sensor(name)
        sensor.track(vehicle.origin, rig["pos"], rig["hpr"])


<<<<<<< HEAD
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
=======
def get_frame(env, name):
    """Return the current RGB frame (uint8, BGR-for-cv2) for a named camera."""
    sensor = env.engine.get_sensor(name)
    frame = sensor.get_rgb_array_cpu()
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def save_all_frames(env, step, out_dir, stereo_matcher):
    frames = {name: get_frame(env, name) for name in CAMERA_RIGS}

    for name, frame in frames.items():
        fname = os.path.join(out_dir, name, f"frame_{step:05d}.png")
        cv2.imwrite(fname, frame)

    # Stereo depth from the front pair
    depth = compute_depth(frames["front_left_camera"], frames["front_right_camera"], stereo_matcher)
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, "front_depth", f"frame_{step:05d}.png"), depth_vis)
    np.save(os.path.join(out_dir, "front_depth", f"frame_{step:05d}.npy"), depth)  # raw metric depth


def main(steps=2000, save_every=10, out_dir="./captured_frames_stereo"):
    for name in list(CAMERA_RIGS.keys()) + ["front_depth"]:
>>>>>>> camera
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)

    env = build_env()
    obs, info = env.reset()
    mount_cameras(env)
<<<<<<< HEAD

    print("=" * 60)
    print("Multi-Camera MetaDrive Capture: front / rear / left / right")
=======
    stereo_matcher = build_stereo_matcher()

    print("=" * 60)
    print("MetaDrive Capture: front-stereo (L/R) / left / right / rear")
>>>>>>> camera
    print("Drive with Arrow Keys or WASD in the MetaDrive window.")
    print("Close the window (or Ctrl+C here) to stop.")
    print("=" * 60)

    try:
        for step in range(steps):
<<<<<<< HEAD
            # action is ignored by the sim while manual_control=True; keys drive it directly
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

            if step % save_every == 0:
                save_all_frames(env, step, out_dir)
                print(f"[step {step}] saved frames from all 4 cameras  reward={reward:.3f}")
=======
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

            if step % save_every == 0:
                save_all_frames(env, step, out_dir, stereo_matcher)
                print(f"[step {step}] saved 5 camera frames + front depth map  reward={reward:.3f}")
>>>>>>> camera

            if terminated or truncated:
                print(f"Episode ended at step {step}, resetting.")
                obs, info = env.reset()
<<<<<<< HEAD
                mount_cameras(env)  # vehicle node is recreated on reset, so re-attach cameras
=======
                mount_cameras(env)
>>>>>>> camera

    except KeyboardInterrupt:
        print("\nStopped by user.")

    env.close()
<<<<<<< HEAD
    print(f"Done. Frames saved under: {out_dir}")


if __name__ == "__main__":
    main()
=======
    print(f"Done. Frames + depth saved under: {out_dir}")


if __name__ == "__main__":
    main()
>>>>>>> camera
