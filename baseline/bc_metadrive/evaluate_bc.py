import numpy as np
import torch

from metadrive import MetaDriveEnv
from baseline.bc_metadrive.model import BCPolicy


# =====================================================
# Device
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =====================================================
# Environment
#
# IMPORTANT:
# Keep this configuration consistent with collect_demos.py.
# manual_control=False because the trained BC policy drives.
# =====================================================

config = dict(
    use_render=True,
    manual_control=False,

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
        show_line_to_navi_mark=False,
    ),

    map=4,
    start_seed=10,
)

env = MetaDriveEnv(config)


# =====================================================
# Load trained BC model
# =====================================================

model = BCPolicy().to(DEVICE)

state_dict = torch.load(
    "checkpoints/bc_policy_50eps.pth",
    map_location=DEVICE,
)

model.load_state_dict(state_dict)

model.eval()

print("\n===================================")
print("Behavioral Cloning Policy Loaded!")
print("===================================\n")


# =====================================================
# Reset
#
# episode_001 was collected starting with seed 10,
# so use seed 21 for this first sanity check.
# =====================================================

obs, info = env.reset(seed=10)

print("Observation type :", type(obs))
print("Observation shape:", obs.shape)
print("Observation min  :", np.min(obs))
print("Observation max  :", np.max(obs))
print("First 10 values  :", obs[:10])
print()


# =====================================================
# Evaluation loop
# =====================================================

done = False
step = 0

previous_obs = obs.copy()

try:

    while not done:

        # ---------------------------------------------
        # Convert current MetaDrive observation
        # into a PyTorch tensor.
        #
        # (259,) -> (1, 259)
        # ---------------------------------------------

        observation_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)


        # ---------------------------------------------
        # BC policy predicts:
        #
        # [steering, throttle/brake]
        # ---------------------------------------------

        with torch.no_grad():
            action_tensor = model(observation_tensor)

        raw_action = (
            action_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )


        # ---------------------------------------------
        # MetaDrive's action range is [-1, 1].
        #
        # Our final network layer is unconstrained,
        # so predictions can slightly exceed this
        # range (for example -1.026).
        #
        # Clip before sending the action to MetaDrive.
        # ---------------------------------------------

        action = np.clip(
            raw_action,
            -1.0,
            1.0,
        )


        # ---------------------------------------------
        # Execute predicted action
        # ---------------------------------------------

        next_obs, reward, terminated, truncated, info = env.step(
            action
        )

        step += 1


        # ---------------------------------------------
        # Debug: did the observation change?
        # ---------------------------------------------

        observation_change = np.mean(
            np.abs(next_obs - previous_obs)
        )


        # ---------------------------------------------
        # Vehicle speed
        # ---------------------------------------------

        speed = env.agent.speed_km_h


        # ---------------------------------------------
        # Print policy behaviour
        # ---------------------------------------------

        print(
            f"Step {step:4d} | "
            f"Steer {action[0]:7.3f} | "
            f"Throttle {action[1]:7.3f} | "
            f"Speed {speed:7.2f} km/h | "
            f"Obs Δ {observation_change:.6f}"
        )

        # Print separately if clipping actually happened
        if not np.allclose(raw_action, action):

            print(
                f"           Raw output was "
                f"[{raw_action[0]:.3f}, "
                f"{raw_action[1]:.3f}] "
                f"-> clipped to "
                f"[{action[0]:.3f}, "
                f"{action[1]:.3f}]"
            )


        # ---------------------------------------------
        # Prepare next timestep
        # ---------------------------------------------

        previous_obs = next_obs.copy()
        obs = next_obs

        done = terminated or truncated

        env.render()


    # =================================================
    # Episode summary
    # =================================================

    print("\n========== Evaluation Complete ==========")

    print(f"Steps          : {step}")
    print(f"Final Speed    : {env.agent.speed_km_h:.2f} km/h")

    if info.get("arrive_dest", False):

        print("Result         : Reached Destination")

    elif info.get("crash_vehicle", False):

        print("Result         : Crashed into Vehicle")

    elif info.get("crash_object", False):

        print("Result         : Hit Road Object")

    elif info.get("out_of_road", False):

        print("Result         : Went Off Road")

    elif info.get("max_step", False):

        print("Result         : Maximum Steps Reached")

    else:

        print("Result         : Episode Ended")


finally:

    env.close()