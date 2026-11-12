"""BC -> PPO compatibility checks (needs torch + stable-baselines3; no simulator)."""

import unittest
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from bev_policy.model import BEVScalarPolicy
from bev_rl.policy import BEVActorCritic, init_from_bc, load_bc_checkpoint

CKPT = Path(__file__).resolve().parents[1] / "bev_policy" / "checkpoints" / "bc_clear_5eps_without_traffic.pt"
OBS = spaces.Dict({"bev": spaces.Box(-1, 1, (1, 64, 64), np.float32),
                   "scalar": spaces.Box(-np.inf, np.inf, (6,), np.float32)})
ACT = spaces.Box(-1, 1, (2,), np.float32)


def build():
    policy = BEVActorCritic(OBS, ACT, lambda _: 3e-4, log_std_init=-1.6)
    ckpt = load_bc_checkpoint(CKPT)
    bc = init_from_bc(policy, ckpt)
    return policy.eval(), bc.eval(), ckpt


class TestBCCompatibility(unittest.TestCase):
    def test_checkpoint_contract(self):
        ckpt = load_bc_checkpoint(CKPT)
        self.assertEqual(ckpt["in_channels"], 1)
        self.assertEqual(len(ckpt["scalar_mean"]), 6)
        self.assertEqual(len(ckpt["scalar_std"]), 6)

    def test_ppo_mean_equals_bc_action_at_initialisation(self):
        policy, bc, _ = build()
        g = torch.Generator().manual_seed(0)
        obs = {"bev": torch.rand(32, 1, 64, 64, generator=g) * 2 - 1, "scalar": torch.randn(32, 6, generator=g)}
        with torch.no_grad():
            bc_action = bc(obs["bev"], obs["scalar"])
            mean = policy.get_distribution(obs).distribution.mean
            deterministic, _ = policy.predict(obs, deterministic=True)
        self.assertTrue(torch.allclose(bc_action, mean, atol=1e-6))
        np.testing.assert_allclose(deterministic, bc_action.numpy(), atol=1e-6)

    def test_shapes_and_value_head(self):
        policy, _, _ = build()
        obs = {"bev": torch.zeros(4, 1, 64, 64), "scalar": torch.zeros(4, 6)}
        with torch.no_grad():
            actions, values, log_prob = policy(obs)
        self.assertEqual(tuple(actions.shape), (4, 2))
        self.assertEqual(tuple(values.shape), (4, 1))
        self.assertEqual(tuple(log_prob.shape), (4,))
        self.assertAlmostEqual(float(policy.log_std.exp()[0]), float(np.exp(-1.6)), places=5)

    def test_value_loss_does_not_touch_actor_parameters(self):
        policy, _, _ = build()
        policy.train()
        obs = {"bev": torch.rand(8, 1, 64, 64), "scalar": torch.randn(8, 6)}
        values, _, _ = policy.evaluate_actions(obs, torch.zeros(8, 2))
        values.sum().backward()
        actor_params = list(policy.pi_features_extractor.parameters()) + list(policy.mlp_extractor.policy_net.parameters()) \
            + list(policy.action_net.parameters())
        self.assertTrue(all(p.grad is None for p in actor_params))
        self.assertTrue(any(p.grad is not None for p in policy.vf_features_extractor.parameters()))

    def test_incompatible_checkpoint_is_rejected(self):
        policy = BEVActorCritic(OBS, ACT, lambda _: 3e-4)
        bad = dict(load_bc_checkpoint(CKPT))
        state = dict(bad["model_state"])
        state["bev_encoder.0.weight"] = torch.zeros(32, 7, 5, 5)
        bad["model_state"] = state
        with self.assertRaises(Exception):
            init_from_bc(policy, bad)


if __name__ == "__main__":
    unittest.main()
