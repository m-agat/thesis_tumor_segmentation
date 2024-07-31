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
from UNet import BuildUNet, BrainTumorDatasetNifti

# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
# ])
transform = None
# image_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
image_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data_subset"
dataset = BrainTumorDatasetNifti(image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize TensorBoard writer
writer = SummaryWriter()

# Model, criterion, optimizer
model = BuildUNet()
criterion = nn.BCEWithLogitsLoss()
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
        loss = criterion(outputs, labels)
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
torch.save(model.state_dict(), "unet_model.pth")

# Close the TensorBoard writer
writer.close()