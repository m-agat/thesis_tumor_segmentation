import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add parent directory to the system path
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from brats_dataset.brats_test_loader import BrainTumorValidationDatasetNifti
from UNet import UNet3D

# Path to your validation dataset
validation_image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-ValidationData/validation_data"

# Initialize dataset and dataloader
validation_dataset = BrainTumorValidationDatasetNifti(image_dir=validation_image_dir)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=4, out_channels=5)
model.load_state_dict(torch.load("unet_model.pth"))
model.to(device)
model.eval()

# Create a directory to save predictions
predictions_dir = "UNet_predictions"
os.makedirs(predictions_dir, exist_ok=True)

# Define a color map for different classes
color_map = {
    0: (0, 0, 0),       # background
    1: (255, 0, 0),     # class 1
    2: (0, 255, 0),     # class 2
    3: (0, 0, 255),     # class 3
    4: (255, 255, 0)    # class 4
}

# Function to apply color map
def apply_color_map(segmentation, color_map):
    h, w = segmentation.shape
    color_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        color_segmentation[segmentation == cls] = color
    return color_segmentation

# Run inference
with torch.no_grad():
    for batch_data in tqdm(validation_dataloader, desc="Validating"):
        inputs = batch_data["image"].to(device)
        case = batch_data["case"][0]
        start_slice = batch_data["start_slice"].item()

        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1).cpu().numpy().squeeze()

        # Iterate over the depth of the volume to save each slice
        for i in range(outputs.shape[0]):
            slice_idx = start_slice + i
            segmentation = outputs[i]

            color_segmentation = apply_color_map(segmentation, color_map)
            plt.imshow(color_segmentation)
            plt.title(f"Case: {case}, Slice: {slice_idx}")
            plt.axis("off")
            plt.savefig(os.path.join(predictions_dir, f"{case}_slice_{slice_idx}.png"))
            plt.close()
