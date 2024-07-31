import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImaged, EnsureChannelFirstd, ToTensord, Compose, SpatialPadd, RandSpatialCropd
from monai.data import DataLoader, Dataset
from glob import glob
from tqdm import tqdm
import torch
from monai.networks.nets import VNet

# Define paths
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
test_image_dir = f"{parent_dir}/BraTS2024-BraTS-GLI-ValidationData/validation_data_subset"

# Get all the case paths
case_paths = glob(os.path.join(test_image_dir, "*"))

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
        "case_id": os.path.basename(case)  # Add case_id to identify the case later
    }
    data_dicts.append(data_dict)

# Define transformations
test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    SpatialPadd(keys=["image"], spatial_size=[128, 128, 128]),  # Padding to ensure consistent dimensions
    RandSpatialCropd(keys=["image"], roi_size=[128, 128, 128], random_size=False),  # Crop to ensure the dimensions are divisible by the pooling factor
    ToTensord(keys=["image"]),
])

# Create dataset and dataloader
test_ds = Dataset(data=data_dicts, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

# Define the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VNet(spatial_dims=3, in_channels=4, out_channels=1).to(device)

# Load the trained model weights
model.load_state_dict(torch.load("vnet_model.pth"))
model.eval()  # Set model to evaluation mode

# Create directories to save predictions
predictions_dir = "VNet_predictions"
predictions_png_dir = "VNet_predictions_png"
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(predictions_png_dir, exist_ok=True)

def save_slices_as_pngs(pred, case_id, predictions_png_dir):
    os.makedirs(os.path.join(predictions_png_dir, case_id), exist_ok=True)
    for i in range(pred.shape[0]):
        slice_img = pred[i, :, :]
        plt.imshow(slice_img, cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(predictions_png_dir, case_id, f'slice_{i}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

with torch.no_grad():
    for batch_data in tqdm(test_loader, desc="Testing"):
        inputs = batch_data["image"].to(device)

        # Run inference
        outputs = model(inputs)

        # Process and save the predictions
        for i in range(outputs.shape[0]):
            pred = outputs[i, 0].cpu().numpy()  # Assuming single-channel output

            # Generate a filename for the prediction using the case_id
            case_id = batch_data['case_id'][i]
            pred_path = os.path.join(predictions_dir, f"{case_id}_pred.nii.gz")

            # Save the prediction as NIfTI
            nib.save(nib.Nifti1Image(pred, np.eye(4)), pred_path)

            # Save the prediction slices as PNGs
            save_slices_as_pngs(pred, case_id, predictions_png_dir)

print("Predictions have been saved.")
