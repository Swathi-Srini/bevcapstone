import os
import glob
import numpy as np

DATA_DIR = "data/train"

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))

print("=" * 60)
print(f"Found {len(files)} episodes")
print("=" * 60)

total_samples = 0
episode_lengths = []

for file in files:
    data = np.load(file)

    obs = data["observation"]
    act = data["action"]

    episode_lengths.append(len(obs))
    total_samples += len(obs)

print(f"Total samples : {total_samples}")
print(f"Average episode length : {np.mean(episode_lengths):.2f}")
print(f"Shortest episode : {np.min(episode_lengths)}")
print(f"Longest episode : {np.max(episode_lengths)}")

print("\n" + "=" * 60)
print("Inspecting first episode")
print("=" * 60)

sample = np.load(files[0])

print("Keys:")
print(sample.files)

obs = sample["observation"]
act = sample["action"]

print("\nObservation")
print("--------------------")
print("Shape :", obs.shape)
print("dtype :", obs.dtype)

print("\nAction")
print("--------------------")
print("Shape :", act.shape)
print("dtype :", act.dtype)

print("\nFirst observation:")
print(obs[0])

print("\nFirst action:")
print(act[0])

print("\nObservation statistics")
print("--------------------")
print("Min :", obs.min())
print("Max :", obs.max())
print("Mean:", obs.mean())
print("Std :", obs.std())

print("\nAction statistics")
print("--------------------")
print("Min :", act.min(axis=0))
print("Max :", act.max(axis=0))
print("Mean:", act.mean(axis=0))
print("Std :", act.std(axis=0))