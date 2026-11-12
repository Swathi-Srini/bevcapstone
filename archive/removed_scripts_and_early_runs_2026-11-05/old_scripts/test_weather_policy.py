"""Weather-conditioned BEV v2 policy checks (torch + SB3 only; no simulator, no checkpoint files)."""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from bev_policy.model import BEVScalarPolicy
from bev_rl.policy import BEVActorCritic, init_from_bc
from bev_state.weather import WEATHER_KINDS, weather_vector

ACT = spaces.Box(-1, 1, (2,), np.float32)


def obs_space(weather: bool):
    d = {"bev": spaces.Box(0, 1, (7, 64, 64), np.float32), "scalar": spaces.Box(-np.inf, np.inf, (6,), np.float32)}
    if weather:
        d["weather"] = spaces.Box(0, 1, (4,), np.float32)
    return spaces.Dict(d)


def batch(n=16, weather=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = {"bev": (torch.rand(n, 7, 64, 64, generator=g) > 0.5).float(), "scalar": torch.randn(n, 6, generator=g)}
    if weather:
        o["weather"] = torch.stack([torch.from_numpy(weather_vector(k, i)) for k, i in
                                    zip(["clear", "fog", "rain", "fog"] * n, [0, .3, .6, 1.0] * n)][:n])
    return o


def make(weather: bool):
    torch.manual_seed(1)
    bc = BEVScalarPolicy(7, 6, 4 if weather else 0).eval()
    ckpt = {"in_channels": 7, "weather_dim": 4 if weather else 0, "model_state": bc.state_dict()}
    policy = BEVActorCritic(obs_space(weather), ACT, lambda _: 3e-4, log_std_init=-1.6)
    init_from_bc(policy, ckpt)
    return policy.eval(), bc


class WeatherVectorTests(unittest.TestCase):
    def test_encoding(self):
        np.testing.assert_array_equal(weather_vector("clear", 0.7), [1, 0, 0, 0])   # clear ignores intensity
        np.testing.assert_array_equal(weather_vector("fog", 0.5), [0, 1, 0, 0.5])
        np.testing.assert_array_equal(weather_vector("rain", 2.0), [0, 0, 1, 1.0])  # clipped
        self.assertEqual(WEATHER_KINDS[:3], ("clear", "fog", "rain")) if isinstance(WEATHER_KINDS, tuple) else None


class PolicyTests(unittest.TestCase):
    def test_ppo_mean_equals_bc_action_with_weather(self):
        policy, bc = make(True)
        o = batch()
        with torch.no_grad():
            ref = bc(o["bev"], o["scalar"], o["weather"])
            mean = policy.get_distribution(o).distribution.mean
        self.assertTrue(torch.allclose(ref, mean, atol=1e-6))

    def test_ppo_mean_equals_bc_action_without_weather(self):
        policy, bc = make(False)
        o = batch(weather=False)
        with torch.no_grad():
            self.assertTrue(torch.allclose(bc(o["bev"], o["scalar"]), policy.get_distribution(o).distribution.mean, atol=1e-6))

    def test_weather_input_changes_output_only_when_present(self):
        policy, _ = make(True)
        o = batch()
        o2 = dict(o, weather=torch.zeros_like(o["weather"]))
        with torch.no_grad():
            a = policy.get_distribution(o).distribution.mean
            b = policy.get_distribution(o2).distribution.mean
        self.assertGreater(float((a - b).abs().max()), 1e-6)   # the branch is really wired in (not dead)

    def test_value_loss_does_not_touch_actor(self):
        policy, _ = make(True)
        policy.train()
        o = batch(8)
        values, _, _ = policy.evaluate_actions(o, torch.zeros(8, 2))
        values.sum().backward()
        actor = list(policy.pi_features_extractor.parameters()) + list(policy.mlp_extractor.policy_net.parameters()) \
            + list(policy.action_net.parameters())
        self.assertTrue(all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in actor))


if __name__ == "__main__":
    unittest.main()
