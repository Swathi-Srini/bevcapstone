import numpy as np
import glob

actions = []

for file in glob.glob("data/train/*.npz"):
    data = np.load(file)
    actions.append(data["action"])

actions = np.concatenate(actions, axis=0)

print("Shape:", actions.shape)

print("Steering")
print("  Min :", actions[:,0].min())
print("  Max :", actions[:,0].max())
print("  Mean:", actions[:,0].mean())

print()

print("Throttle")
print("  Min :", actions[:,1].min())
print("  Max :", actions[:,1].max())
print("  Mean:", actions[:,1].mean())