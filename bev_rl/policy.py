"""Actor-critic with the intended architecture, initialised from the BC checkpoint.

    BEV (1,64,64) --CNN--> 128 \
                                 concat(160) --actor head--> tanh mean (2) + learned log-std
    scalar (6)    --MLP--> 32  /           \--critic head--> V(s)

The actor's feature extractor and head are *exactly* the committed BC modules
(`bev_policy.model.BEVScalarPolicy`), so at initialisation the PPO policy mean equals
the BC action for every input (verified in tests/test_bev_rl_policy.py). The critic has
its own encoder (``share_features_extractor=False``) so value-loss gradients cannot
disturb the BC-initialised actor features; the critic encoder is also initialised from
the BC encoder weights (a design choice, not a requirement).

Adaptations required for PPO (BC is deterministic):
  * a state-independent learnable log-std turns the deterministic head into a Gaussian;
  * the tanh on the mean is kept (as in BC), the sampled action is clipped to [-1, 1] by SB3;
  * a randomly initialised critic head is added.
"""

from __future__ import annotations

from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from bev_policy.model import BEVScalarPolicy


class BEVFeatures(BaseFeaturesExtractor):
    """BEV CNN + 6-D MLP (+ optional 4-D weather encoder). Channel/weather sizes are read from the observation space."""

    def __init__(self, observation_space: spaces.Dict, scalar_dim: int = 6):
        in_channels = int(observation_space["bev"].shape[0])
        weather_dim = int(observation_space["weather"].shape[0]) if "weather" in observation_space.spaces else 0
        super().__init__(observation_space, features_dim=128 + 32 + (16 if weather_dim else 0))
        base = BEVScalarPolicy(in_channels, scalar_dim, weather_dim)
        self.bev_encoder, self.scalar_encoder, self.weather_encoder = base.bev_encoder, base.scalar_encoder, base.weather_encoder

    def forward(self, obs):
        feats = [self.bev_encoder(obs["bev"]), self.scalar_encoder(obs["scalar"])]
        if self.weather_encoder is not None:
            feats.append(self.weather_encoder(obs["weather"]))
        return torch.cat(feats, dim=1)


class BEVActorCritic(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs.update(net_arch=dict(pi=[128], vf=[128]), activation_fn=nn.ReLU, ortho_init=False,
                      features_extractor_class=BEVFeatures, share_features_extractor=False)
        super().__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        mean = torch.tanh(self.action_net(latent_pi))      # same output nonlinearity as BC
        return self.action_dist.proba_distribution(mean, self.log_std)


def load_bc_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ckpt.setdefault("weather_dim", 0)
    return ckpt


def init_from_bc(policy: BEVActorCritic, ckpt: dict, init_critic_encoder: bool = True) -> BEVScalarPolicy:
    """Copy BC weights into the actor (and optionally the critic encoder). Returns the BC module."""
    bc = BEVScalarPolicy(int(ckpt["in_channels"]), 6, int(ckpt.get("weather_dim", 0)))
    bc.load_state_dict(ckpt["model_state"])                 # raises on any shape mismatch
    pi_fe = policy.pi_features_extractor
    pi_fe.bev_encoder.load_state_dict(bc.bev_encoder.state_dict())
    pi_fe.scalar_encoder.load_state_dict(bc.scalar_encoder.state_dict())
    if bc.weather_encoder is not None:
        pi_fe.weather_encoder.load_state_dict(bc.weather_encoder.state_dict())
    if init_critic_encoder:
        vf_fe = policy.vf_features_extractor
        vf_fe.bev_encoder.load_state_dict(bc.bev_encoder.state_dict())
        vf_fe.scalar_encoder.load_state_dict(bc.scalar_encoder.state_dict())
        if bc.weather_encoder is not None:
            vf_fe.weather_encoder.load_state_dict(bc.weather_encoder.state_dict())
    policy.mlp_extractor.policy_net[0].load_state_dict(bc.action_head[0].state_dict())   # Linear(160[+16],128)
    policy.action_net.load_state_dict(bc.action_head[2].state_dict())                    # Linear(128,2)
    return bc
