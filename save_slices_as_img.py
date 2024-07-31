import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the path to the validation data and the output directory for images
validation_image_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-ValidationData/validation_data_subset"
output_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-ValidationData/slice_images"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get all the case paths
case_paths = os.listdir(validation_image_dir)

# Function to save slices as images
def save_slices_as_images(case_path, output_dir):
    case_name = os.path.basename(case_path)
    modalities = ["t1c", "t1n", "t2f", "t2w"]

    # Load each modality
    for modality in modalities:
        modality_path = os.path.join(case_path, f"{case_name}-{modality}.nii.gz")
        if not os.path.exists(modality_path):
            continue

        # Load the NIfTI file
        img = nib.load(modality_path)
        data = img.get_fdata()

        # Save each slice as an image
        for slice_idx in range(data.shape[2]):
            slice_data = data[:, :, slice_idx]
            plt.imshow(slice_data, cmap="gray")
            plt.axis("off")
            output_path = os.path.join(output_dir, f"{case_name}_{modality}_slice_{slice_idx}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# Iterate through each case and save slices as images
for case in tqdm(case_paths, desc="Creating slice images"):
    case_path = os.path.join(validation_image_dir, case)
    save_slices_as_images(case_path, output_dir)

print("Slice images have been created and saved.")
