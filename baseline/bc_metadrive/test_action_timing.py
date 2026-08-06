import numpy as np
from metadrive import MetaDriveEnv

env = MetaDriveEnv(
    dict(
        use_render=True,
        manual_control=True,
    )
)

obs, info = env.reset()
env.agent.expert_takeover = True

print("Press T to take over.")
print("Drive for a few seconds.\n")

for i in range(20):

    print(f"\n------ STEP {i} ------")

    print("Before step :", np.array(env.agent.current_action))

    obs, reward, terminated, truncated, info = env.step([0, 0])

    print("After step  :", np.array(env.agent.current_action))

    env.render()

env.close()