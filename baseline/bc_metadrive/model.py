import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy Network

    Input:
        Observation vector (259)

    Output:
        Action vector (2)
        [steering, throttle]
    """

    def __init__(self,
                 observation_dim=259,
                 hidden_dim=256,
                 action_dim=2):

        super().__init__()

        # -------------------------
        # Feature Extractor
        # -------------------------
        self.feature_extractor = nn.Sequential(

            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # -------------------------
        # Policy Head
        # -------------------------
        self.policy_head = nn.Sequential(

            nn.Linear(hidden_dim, 128),
            nn.ReLU(),

            nn.Linear(128, action_dim)
        )

    def forward(self, observation):

        features = self.feature_extractor(observation)

        action = self.policy_head(features)

        return action