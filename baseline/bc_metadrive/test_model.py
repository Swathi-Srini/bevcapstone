import torch
from baseline.bc_metadrive.model import BCPolicy

# Create model
model = BCPolicy()

# Dummy batch
dummy_input = torch.randn(64, 259)

# Forward pass
output = model(dummy_input)

print(model)
print()

print("Input shape :", dummy_input.shape)
print("Output shape:", output.shape)