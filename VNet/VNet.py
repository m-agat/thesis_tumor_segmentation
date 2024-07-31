import os
import nibabel as nib
import numpy as np
from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord, Compose, SpatialPadd, RandSpatialCropd
from monai.data import DataLoader, CacheDataset
from monai.utils import set_determinism
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.networks.nets import VNet
from monai.losses import DiceLoss

# Ensure reproducibility
set_determinism(seed=0)

# Define paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-TrainingData/training_data_subset"

# Get all the case paths
case_paths = glob(os.path.join(image_dir, "*"))

# Create a list of dictionaries for the dataset
data_dicts = []
for case in case_paths:
    data_dict = {
        "image": [
            os.path.join(case, f"{os.path.basename(case)}-t1c.nii.gz"),
            os.path.join(case, f"{os.path.basename(case)}-t1n.nii.gz"),
            os.path.join(case, f"{os.path.basename(case)}-t2f.nii.gz"),
            os.path.join(case, f"{os.path.basename(case)}-t2w.nii.gz"),
        ],
        "label": os.path.join(case, f"{os.path.basename(case)}-seg.nii.gz"),
    }
    data_dicts.append(data_dict)

# Define transformations with padding and cropping
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    SpatialPadd(keys=["image", "label"], spatial_size=[128, 128, 128]),  # Padding to ensure consistent dimensions
    RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),  # Crop to ensure the dimensions are divisible by the pooling factor
    ToTensord(keys=["image", "label"]),
])

# Create dataset and dataloader
train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)  # Reduced batch size

# Initialize TensorBoard writer
writer = SummaryWriter()

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
model = VNet(spatial_dims=3, in_channels=4, out_channels=1).to(device)

# Define the loss function and optimizer
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Log loss to TensorBoard
        writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + batch_idx)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item()}")

    # Log average loss per epoch to TensorBoard
    writer.add_scalar("Average Loss per Epoch", epoch_loss / len(train_loader), epoch)

    print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {epoch_loss / len(train_loader)}")

print("Training Complete")
torch.save(model.state_dict(), "vnet_model.pth")

# Close the TensorBoard writer
writer.close()
