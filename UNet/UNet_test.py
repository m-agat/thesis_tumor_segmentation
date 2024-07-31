import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from UNet import BuildUNet


class BrainTumorValidationDatasetNifti(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.cases = os.listdir(image_dir)

    def __len__(self):
        return len(self.cases) * self._get_num_slices()

    def _get_num_slices(self):
        example_case = os.path.join(self.image_dir, self.cases[0])
        example_nii = nib.load(os.path.join(example_case, os.listdir(example_case)[0]))
        return example_nii.shape[2]

    def __getitem__(self, idx):
        case_idx = idx // self._get_num_slices()
        slice_idx = idx % self._get_num_slices()
        
        case = self.cases[case_idx]
        case_dir = os.path.join(self.image_dir, case)
        
        t1c = nib.load(os.path.join(case_dir, f"{case}-t1c.nii.gz")).get_fdata()[:, :, slice_idx]
        t1n = nib.load(os.path.join(case_dir, f"{case}-t1n.nii.gz")).get_fdata()[:, :, slice_idx]
        t2f = nib.load(os.path.join(case_dir, f"{case}-t2f.nii.gz")).get_fdata()[:, :, slice_idx]
        t2w = nib.load(os.path.join(case_dir, f"{case}-t2w.nii.gz")).get_fdata()[:, :, slice_idx]
        
        # Stack the modalities to create a multi-channel input
        image = np.stack([t1c, t1n, t2f, t2w], axis=0)
        
        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).float()

        return {"image": image, "case": case, "slice_idx": slice_idx}

# Path to your test dataset
transform = False
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
validation_image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-ValidationData/validation_data_subset"
validation_dataset = BrainTumorValidationDatasetNifti(image_dir=validation_image_dir, transform=transform)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Load the trained model
model = BuildUNet()
model.load_state_dict(torch.load("unet_model.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

import matplotlib.pyplot as plt
import os

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BuildUNet()
model.load_state_dict(torch.load("unet_model.pth"))
model.to(device)
model.eval()

# Create a directory to save predictions
predictions_dir = "UNet_predictions"
os.makedirs(predictions_dir, exist_ok=True)

# Run inference
with torch.no_grad():
    for batch_data in tqdm(validation_dataloader, desc="Validating"):
        inputs = batch_data["image"].to(device)
        case = batch_data["case"][0]
        slice_idx = batch_data["slice_idx"].item()

        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().numpy().squeeze()

        # Save the prediction
        plt.imshow(outputs, cmap="gray")
        plt.title(f"Case: {case}, Slice: {slice_idx}")
        plt.axis("off")
        plt.savefig(os.path.join(predictions_dir, f"{case}_slice_{slice_idx}.png"))
        plt.close()

