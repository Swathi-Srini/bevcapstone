import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from bev_policy.dataset import BEVDataset, episode_files, load_episode, split_files
from bev_policy.model import BEVScalarPolicy
from bev_policy.train_bc import action_loss, main


def write_episode(directory: Path, seed: int, include_braking: bool = True) -> Path:
    count = 4
    scalar = np.arange(count * 6, dtype=np.float32).reshape(count, 6) + seed
    action = np.tile(np.array([[0.1, 0.4], [-0.2, -0.5]], dtype=np.float32), (2, 1))
    if not include_braking:
        action[:, 1] = np.abs(action[:, 1])
    bev = np.full((count, 64, 64), seed / 100.0, dtype=np.float32)
    path = directory / f"seed_{seed:05d}.npz"
    np.savez_compressed(path, bev=bev, scalar=scalar, action=action)
    return path


class TestBEVBehavioralCloning(unittest.TestCase):
    def test_episode_split_and_scalar_normalization_are_seed_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            files = [write_episode(directory, seed) for seed in (11, 12, 13, 14, 15)]
            train_files, validation_files = split_files(episode_files(directory))

            self.assertFalse(set(train_files) & set(validation_files))
            self.assertEqual(len(train_files) + len(validation_files), len(files))
            train = BEVDataset(train_files)
            validation = BEVDataset(validation_files, train.mean, train.std)
            raw_validation = load_episode(validation_files[0])[1]
            np.testing.assert_allclose(
                validation.scalar[:len(raw_validation)],
                (raw_validation - train.mean) / train.std,
            )

    def test_episode_loader_rejects_misaligned_action_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seed_00011.npz"
            np.savez(path, bev=np.zeros((2, 64, 64)), scalar=np.zeros((2, 6)), action=np.zeros((1, 2)))
            with self.assertRaises(ValueError):
                load_episode(path)

    def test_policy_outputs_bounded_two_control_actions(self):
        policy = BEVScalarPolicy()
        actions = policy(torch.zeros(3, 1, 64, 64), torch.zeros(3, 6))
        self.assertEqual(tuple(actions.shape), (3, 2))
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))

    def test_spatial_bev_encoder_preserves_grid_features(self):
        policy = BEVScalarPolicy(bev_pool_size=4)
        self.assertEqual(policy.bev_pool_size, 4)
        self.assertEqual(policy.bev_encoder[8].in_features, 96 * 4 * 4)
        actions = policy(torch.zeros(2, 1, 64, 64), torch.zeros(2, 6))
        self.assertEqual(tuple(actions.shape), (2, 2))

    def test_brake_weight_increases_loss_only_for_braking_targets(self):
        prediction = torch.zeros(2, 2)
        target = torch.tensor([[0.0, -0.04], [0.0, -0.06]])
        weighted_loss = action_loss(prediction, target, 3.0)
        ordinary_loss = action_loss(prediction, target, 1.0)
        self.assertGreater(weighted_loss, ordinary_loss)
        torch.testing.assert_close(weighted_loss, torch.tensor((0.04**2 + 3 * 0.06**2) / 4))

    def test_training_saves_seeded_checkpoint_with_episode_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for seed in (11, 12, 13):
                write_episode(data_dir, seed)
            checkpoints = [root / "first.pt", root / "second.pt"]
            for checkpoint in checkpoints:
                args = [
                    "train_bc.py", "--data-dir", str(data_dir), "--checkpoint", str(checkpoint),
                    "--epochs", "2", "--batch-size", "4", "--patience", "2", "--seed", "17",
                ]
                with patch("sys.argv", args):
                    self.assertEqual(main(), 0)

            first = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
            second = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
            self.assertEqual(first["seed"], 17)
            self.assertEqual(first["epochs_trained"], 2)
            self.assertIn("best_val_loss", first)
            self.assertNotIn("best_val_mse", first)
            self.assertFalse(set(first["train_episodes"]) & set(first["validation_episodes"]))
            self.assertEqual(first["train_episodes"], second["train_episodes"])
            self.assertEqual(first["validation_episodes"], second["validation_episodes"])
            for name, value in first["model_state"].items():
                torch.testing.assert_close(value, second["model_state"][name], rtol=0, atol=0)

    def test_training_quality_gate_rejects_missing_braking_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for seed in (11, 12, 13):
                write_episode(data_dir, seed, include_braking=False)
            args = [
                "train_bc.py", "--data-dir", str(data_dir),
                "--checkpoint", str(root / "unused.pt"), "--min-brake-samples", "1",
            ]
            with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "braking samples"):
                main()

    def test_warm_start_preserves_checkpoint_scalar_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            for seed in (11, 12, 13):
                write_episode(data_dir, seed)
            scalar_mean = np.full(6, 3.0, dtype=np.float32)
            scalar_std = np.full(6, 2.0, dtype=np.float32)
            initial_checkpoint = root / "initial.pt"
            torch.save({
                "model_state": BEVScalarPolicy().state_dict(),
                "in_channels": 1,
                "scalar_mean": scalar_mean.tolist(),
                "scalar_std": scalar_std.tolist(),
            }, initial_checkpoint)
            output_checkpoint = root / "fine_tuned.pt"
            args = [
                "train_bc.py", "--data-dir", str(data_dir),
                "--checkpoint", str(output_checkpoint), "--epochs", "1",
                "--batch-size", "4", "--init-checkpoint", str(initial_checkpoint),
            ]
            with patch("sys.argv", args):
                self.assertEqual(main(), 0)

            trained = torch.load(output_checkpoint, map_location="cpu", weights_only=True)
            np.testing.assert_array_equal(trained["scalar_mean"], scalar_mean)
            np.testing.assert_array_equal(trained["scalar_std"], scalar_std)
            self.assertEqual(trained["bev_pool_size"], 4)


if __name__ == "__main__":
    unittest.main()
