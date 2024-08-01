import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from DenseUNet import DenseUNet3D

# Add parent directory to the system path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from brats_dataset.brats_loader3d import BrainTumorDatasetNifti

# Dataset and DataLoader setup
transform = None
# image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-TrainingData/training_data_subset"
cases_subset = ["BraTS-GLI-00005-100", "BraTS-GLI-02064-102", "BraTS-GLI-02069-102", "BraTS-GLI-02085-103"]
dataset = BrainTumorDatasetNifti(image_dir=image_dir, cases_subset=cases_subset, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Model, criterion, optimizer
model = DenseUNet3D(n_classes=5, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, downsample=False)  # For multi-class segmentation
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze(1))  # CrossEntropyLoss expects shape (N, H, W) or (N, D, H, W)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Log loss to TensorBoard
        writer.add_scalar("Training Loss", loss.item(), epoch * len(dataloader) + batch_idx)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item()}")

    # Log average loss per epoch to TensorBoard
    writer.add_scalar("Average Loss per Epoch", epoch_loss / len(dataloader), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {epoch_loss / len(dataloader)}")

print("Training Complete")
torch.save(model.state_dict(), "denseunet3d_model.pth")

# Close the TensorBoard writer
writer.close()