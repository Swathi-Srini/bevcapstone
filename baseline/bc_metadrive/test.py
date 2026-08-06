from metadrive import MetaDriveEnv

env = MetaDriveEnv(
    dict(
        use_render=True,
        map=4,
        start_seed=10,
        num_scenarios=10,
    )
)

obs, info = env.reset(seed=10)

print(obs.shape)

env.close()