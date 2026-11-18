import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from bev_policy.model import BEVScalarPolicy
from bev_policy.ppo import BEVDrivingEnv, BEVFusionFeatures, build_bc_initialized_ppo


class TinyBEVEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "bev": spaces.Box(-1.0, 1.0, shape=(1, 64, 64), dtype=np.float32),
            "scalar": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"bev": np.zeros((1, 64, 64), dtype=np.float32),
                "scalar": np.zeros(6, dtype=np.float32)}, {}

    def step(self, action):
        return ({"bev": np.zeros((1, 64, 64), dtype=np.float32),
                 "scalar": np.zeros(6, dtype=np.float32)}, 0.0, False, False, {})


class TestBCInitializedPPO(unittest.TestCase):
    def test_feature_contract_and_bc_weight_transfer(self):
        bc = BEVScalarPolicy(bev_pool_size=4)
        with torch.no_grad():
            for parameter in bc.parameters():
                parameter.fill_(0.001)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bc.pt"
            torch.save({
                "model_state": bc.state_dict(),
                "in_channels": 1,
                "bev_pool_size": 4,
                "scalar_mean": [0.0] * 6,
                "scalar_std": [1.0] * 6,
            }, checkpoint)
            ppo = build_bc_initialized_ppo(
                TinyBEVEnv(), checkpoint, n_steps=4, batch_size=2
            )

            self.assertTrue(ppo.use_sde)
            self.assertTrue(ppo.policy.squash_output)
            self.assertTrue(all(
                not parameter.requires_grad
                for parameter in ppo._bc_reference.parameters()
            ))
        torch.testing.assert_close(
            ppo.policy.features_extractor.bev_encoder[0].weight,
            bc.bev_encoder[0].weight,
        )
        self.assertEqual(
            ppo.policy.features_extractor.bev_encoder[8].in_features,
            96 * 4 * 4,
        )
        torch.testing.assert_close(
            ppo.policy.mlp_extractor.policy_net[0].weight,
            bc.action_head[0].weight,
        )
        observation = {
            "bev": np.random.default_rng(42).uniform(
                0.0, 1.0, (2, 1, 64, 64)
            ).astype(np.float32),
            "scalar": np.random.default_rng(43).normal(
                size=(2, 6)
            ).astype(np.float32),
        }
        with torch.no_grad():
            bc_actions = bc(
                torch.from_numpy(observation["bev"]),
                torch.from_numpy(observation["scalar"]),
            ).numpy()
        ppo_actions, _ = ppo.predict(observation, deterministic=True)
        np.testing.assert_allclose(ppo_actions, bc_actions, atol=1e-6)

        with torch.no_grad():
            ppo.policy.action_net.bias[0].add_(0.1)
        anchor_loss = ppo._bc_anchor_loss({
            key: torch.from_numpy(value) for key, value in observation.items()
        })
        anchor_loss.backward()
        self.assertGreater(float(ppo.policy.action_net.bias.grad[0]), 0.0)
        action, _ = ppo.predict({
            "bev": np.zeros((1, 64, 64), dtype=np.float32),
            "scalar": np.zeros(6, dtype=np.float32),
        }, deterministic=True)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        ppo.learn(total_timesteps=4)

    def test_training_resamples_meta_drive_scenario_each_episode(self):
        class SeedRecordingSimulator:
            def __init__(self):
                self.seeds = []

            def reset(self, seed):
                self.seeds.append(seed)
                return None, {}

            def close(self):
                pass

        simulator = SeedRecordingSimulator()
        with patch("bev_policy.ppo.make_env", return_value=simulator):
            env = BEVDrivingEnv(
                object(), np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32),
                seed=2026,
            )
        env.assembler.mount_if_needed = lambda _: None
        env._observation = lambda info: {
            "bev": np.zeros((1, 64, 64), dtype=np.float32),
            "scalar": np.zeros(6, dtype=np.float32),
        }

        try:
            env.reset(seed=2026)
            env.reset()
            env.reset()
        finally:
            env.close()

        self.assertEqual(simulator.seeds[0], 2026)
        self.assertEqual(len(set(simulator.seeds)), 3)
        self.assertTrue(all(10 <= seed < 10010 for seed in simulator.seeds[1:]))


if __name__ == "__main__":
    unittest.main()
