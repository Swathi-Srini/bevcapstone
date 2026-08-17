import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class BehavioralCloningDataset(Dataset):
    """
    Behavioral Cloning Dataset

    Current:
        Observation (259,) -> Action (2,)

    Future:
        BEV (64x64) + Ego State -> Action (2,)
    """

    def __init__(self, data_dir="data/train"):

        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

        if len(files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir}")

        observations = []
        actions = []

        print(f"Loading {len(files)} demonstration files...")

        for file in files:

            data = np.load(file)

            observations.append(data["observation"])
            actions.append(data["action"])

        self.observations = np.concatenate(observations, axis=0).astype(np.float32)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32)

        print("----------------------------------------")
        print("Dataset Loaded")
        print("----------------------------------------")
        print(f"Samples      : {len(self.observations)}")
        print(f"Observation  : {self.observations.shape}")
        print(f"Action       : {self.actions.shape}")
        print("----------------------------------------")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):

        observation = torch.from_numpy(self.observations[index])
        action = torch.from_numpy(self.actions[index])

        return observation, action