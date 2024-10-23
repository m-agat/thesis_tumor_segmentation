import sys

sys.path.append("../")
import models.models as models
import os
import torch
import config.config as config
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

model = models.swinunetr_model

checkpoint = torch.load(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/swinunetr_model.pt"
)
model.load_state_dict(checkpoint["state_dict"])
model.to(config.device)
model.eval()

model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[config.roi[0], config.roi[1], config.roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)


def visualize_slices(
    image_slice, ground_truth_slice, predicted_slice, patient_path, slice_num
):
    plt.figure(figsize=(15, 5))

    # Original image (take one modality for visualization, e.g., FLAIR or T1ce)
    plt.subplot(1, 3, 1)
    plt.imshow(image_slice, cmap="gray")
    plt.title(f"Original Image (Slice {slice_num})")
    plt.axis("off")

    # Ground truth segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_slice)
    plt.title(f"Ground Truth Segmentation (Slice {slice_num})")
    plt.axis("off")

    # Generated segmentation
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_slice)
    plt.title(f"Generated Segmentation (Slice {slice_num})")
    plt.axis("off")

    plt.suptitle(f"Patient: {patient_path}, Slice: {slice_num}", fontsize=16)

    # Save the figure as a file
    plt.savefig(f"/home/agata/Desktop/thesis_tumor_segmentation/figures/testing/visualization_{patient_path}_{slice_num}.png")
    plt.close()


with torch.no_grad():
    for batch_data in config.test_loader:
        image = batch_data["image"].cuda()
        patient_path = batch_data["path"]

        # Extract the affine matrix from the original image
        original_image_path = f"/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/test/{patient_path[0]}/{patient_path[0]}_flair.nii.gz"
        original_nifti = nib.load(original_image_path)
        affine = original_nifti.affine

        # Perform inference
        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4

        # Save the segmentation output as a NIfTI file using the extracted affine
        nifti_img = nib.Nifti1Image(seg_out, affine)
        nib.save(nifti_img, f"/home/agata/Desktop/thesis_tumor_segmentation/results/new/{patient_path[0]}_segmentation.nii.gz")

        # Load ground truth segmentation for comparison
        ground_truth = batch_data["label"]  # Assuming NIfTI format for ground truth

        # Select a slice to visualize (e.g., the middle slice)
        slice_num = 90

        # Visualize one modality of the original image (e.g., FLAIR)
        image_slice = (
            image[0, 0, slice_num].cpu().numpy()
        )  # Assuming the first channel (FLAIR)

        # Ground truth segmentation slice
        ground_truth_slice = (
            ground_truth[0, 0, :, :, slice_num].cpu().numpy()
        )  # Taking the first channel
        predicted_slice = seg[0, :, :, slice_num]

        # Call the visualization function
        visualize_slices(
            image_slice, ground_truth_slice, predicted_slice, patient_path, slice_num
        )
