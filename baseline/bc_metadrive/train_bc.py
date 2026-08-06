import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.bc_metadrive.bc_dataset import BehavioralCloningDataset
from baseline.bc_metadrive.model import BCPolicy


# -------------------------
# Hyperparameters
# -------------------------

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dataset
# -------------------------

dataset = BehavioralCloningDataset()

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -------------------------
# Model
# -------------------------

model = BCPolicy().to(DEVICE)


# -------------------------
# Loss + Optimizer
# -------------------------

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training Loop
# -------------------------

print("\nStarting Behavioral Cloning Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for observations, actions in dataloader:

        observations = observations.to(DEVICE)
        actions = actions.to(DEVICE)

        # -----------------
        # Forward Pass
        # -----------------

        predictions = model(observations)

        loss = criterion(predictions, actions)

        # -----------------
        # Backpropagation
        # -----------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}]"
        f"  Loss: {average_loss:.6f}"
    )


# -------------------------
# Save Model
# -------------------------

os.makedirs("checkpoints", exist_ok=True)

torch.save(
    model.state_dict(),
    "checkpoints/bc_policy.pth"
)

print("\nTraining Complete!")
print("Model saved to checkpoints/bc_policy.pth")