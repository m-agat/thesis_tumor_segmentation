
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the paths
predictions_dir = f"{os.getcwd()}/nnUNet_predictions"
output_dir = f"{os.getcwd()}/nnUNet_val_slice_images"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Function to save slices as images
def save_slices_as_images(prediction, case_name, output_dir):
    for slice_idx in range(prediction.shape[2]):
        slice_data = prediction[:, :, slice_idx]
        plt.imshow(slice_data, cmap="gray")
        plt.axis("off")
        output_path = os.path.join(output_dir, f"{case_name}_slice_{slice_idx}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# Iterate through each prediction file and save slices as images
for case_file in tqdm(os.listdir(predictions_dir), desc="Creating slice images"):
    if case_file.endswith(".nii.gz"):
        case_path = os.path.join(predictions_dir, case_file)
        case_name = os.path.splitext(os.path.splitext(case_file)[0])[0]
        
        # Load the NIfTI file
        prediction = nib.load(case_path).get_fdata()
        
        # Save the slices as images
        save_slices_as_images(prediction, case_name, output_dir)

print("Slice images have been created and saved.")
