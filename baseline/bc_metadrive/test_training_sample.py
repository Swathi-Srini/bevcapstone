import torch

from baseline.bc_metadrive.bc_dataset import BehavioralCloningDataset
from baseline.bc_metadrive.model import BCPolicy


# ---------------------------------
# Device
# ---------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------
# Load Dataset
# ---------------------------------

dataset = BehavioralCloningDataset("data/train")

print(f"Dataset Size: {len(dataset)}")


# ---------------------------------
# Load Model
# ---------------------------------

model = BCPolicy().to(DEVICE)

model.load_state_dict(
    torch.load(
        "checkpoints/bc_policy.pth",
        map_location=DEVICE
    )
)

model.eval()


# ---------------------------------
# Test a few training samples
# ---------------------------------

print("\n========== Testing Training Samples ==========\n")

indices = [0, 100, 1000, 5000, 10000]

with torch.no_grad():

    for idx in indices:

        observation, expert_action = dataset[idx]

        prediction = model(
            observation.unsqueeze(0).to(DEVICE)
        )

        prediction = prediction.squeeze(0).cpu()

        print(f"Sample {idx}")
        print(f"Expert     : {expert_action.numpy()}")
        print(f"Prediction : {prediction.numpy()}")

        error = torch.abs(prediction - expert_action)

        print(f"Abs Error  : {error.numpy()}")
        print("-" * 50)