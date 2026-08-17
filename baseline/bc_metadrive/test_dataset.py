from baseline.bc_metadrive.bc_dataset import BehavioralCloningDataset

dataset = BehavioralCloningDataset()

print()

print("Length:", len(dataset))

obs, action = dataset[0]

print("Observation shape:", obs.shape)
print("Action shape:", action.shape)

print()

print(obs)
print(action)