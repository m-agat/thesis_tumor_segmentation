import os
import nibabel as nib
import numpy as np
from monai.transforms import (
    LoadImaged,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
)
from monai.data import DataLoader, CacheDataset, Dataset
from monai.utils import set_determinism
from glob import glob

set_determinism(seed=0)

# Define paths
# image_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
image_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data_subset"

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

# Define transformations
train_transforms = [
    LoadImaged(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(128, 128, 64),
        pos=1,
        neg=1,
        num_samples=4,
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.50),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.50),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.50),
    RandRotate90d(keys=["image", "label"], prob=0.50, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ToTensord(keys=["image", "label"]),
]

# Create dataset and dataloader
train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, num_workers=4)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

import monai
from monai.networks.nets import VNet
from monai.losses import DiceLoss
import torch

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
model = VNet(spatial_dims=3, in_channels=4, out_channels=1).to(device)

# Define the loss function and optimizer
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from monai.metrics import compute_meandice
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Activations

# Define post-processing transforms
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)

# Define the training loop
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch_data in train_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

print("Training Complete")
torch.save(model.state_dict(), "vnet_model.pth")

