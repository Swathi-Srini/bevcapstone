import os
import argparse
import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE


# =====================================================
# Save folder
# =====================================================

SAVE_DIR = "data/train"

os.makedirs(SAVE_DIR, exist_ok=True)


# =====================================================
# Environment
# =====================================================

config = dict(
    use_render=True,
    manual_control=True,

    traffic_density=0.1,

    num_scenarios=10000,

    random_agent_model=False,
    random_lane_width=True,
    random_lane_num=True,

    on_continuous_line_done=False,
    out_of_route_done=True,

    vehicle_config=dict(
        show_lidar=True,
        show_navi_mark=False,
        show_line_to_navi_mark=False
    ),

    map=4,
    start_seed=10,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--observation",
    type=str,
    default="lidar",
    choices=["lidar", "rgb_camera"]
)

args = parser.parse_args()


if args.observation == "rgb_camera":

    config.update(
        dict(
            image_observation=True,

            sensors=dict(
                rgb_camera=(RGBCamera, 400, 300)
            ),

            interface_panel=[
                "rgb_camera",
                "dashboard"
            ]
        )
    )


env = MetaDriveEnv(config)


# =====================================================
# Recording
# =====================================================

episode = 1

try:

    obs, info = env.reset(seed=21)

    print(HELP_MESSAGE)
    print("\nPress T to take over.")
    print("Drive using WASD.")
    print("Press ESC to quit.\n")

    env.agent.expert_takeover = True

    observations = []
    actions = []

    while True:

        # ------------------------------------------------
        # Save the current state/action pair.
        #
        # The observation returned by env.step() is the next
        # state, so we must store the observation before the
        # step to keep BC labels aligned with the state that
        # produced them.
        # ------------------------------------------------

        observations.append(obs)

        # ------------------------------------------------
        # Step environment
        # ------------------------------------------------

        obs, reward, terminated, truncated, info = env.step([0, 0])

        # ------------------------------------------------
        # Current manual action
        # ------------------------------------------------

        action = np.array(env.agent.current_action)

        actions.append(action)

        # ------------------------------------------------
        # Render
        # ------------------------------------------------

        env.render(
            text={
                "Episode": episode,

                "Auto Drive":
                    "ON" if env.agent.expert_takeover else "OFF",

                "Samples":
                    len(observations),

                "Keyboard":
                    "W A S D",

                "Press":
                    "T to switch control"
            }
        )

        # ------------------------------------------------
        # RGB preview
        # ------------------------------------------------

        if args.observation == "rgb_camera":

            cv2.imshow(
                "RGB",
                obs["image"][..., -1]
            )

            cv2.waitKey(1)

        # ------------------------------------------------
        # End of episode
        # ------------------------------------------------

        if terminated or truncated:

            filename = os.path.join(
                SAVE_DIR,
                f"episode_{episode:03d}.npz"
            )

            np.savez_compressed(

                filename,

                observation=np.array(observations),

                action=np.array(actions)

            )

            print(
                f"Saved {filename} "
                f"({len(observations)} samples)"
            )

            episode += 1

            observations = []
            actions = []

            obs, info = env.reset(
                seed=env.current_seed + 1
            )

            env.agent.expert_takeover = True

finally:

    env.close()