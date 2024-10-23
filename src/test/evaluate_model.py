import nibabel as nib
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.utils.enums import MetricReduction
import os
import numpy as np

# Initialize Dice and Hausdorff metrics
dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction=MetricReduction.MEAN_BATCH)
sensitivity_metric = ConfusionMatrixMetric(metric_name="sensitivity", include_background=True, reduction=MetricReduction.MEAN_BATCH)
specificity_metric = ConfusionMatrixMetric(metric_name="specificity", include_background=True, reduction=MetricReduction.MEAN_BATCH)

pred_dir = "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/SwinUNetr_Results_Test_Set"
gt_dir = "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/test/"

# Reset metrics before computing
dice_metric.reset()
hd95_metric.reset()
sensitivity_metric.reset()
specificity_metric.reset()

# Assuming you have a list of patient IDs
patient_ids = ["BraTS2021_00000", "BraTS2021_00001", "BraTS2021_00002"]  # Replace with actual patient IDs

for patient_id in patient_ids:
    # Load predicted segmentation
    pred_path = os.path.join(pred_dir, f"{patient_id}_segmentation.nii.gz")
    pred_nifti = nib.load(pred_path)
    pred_data = torch.tensor(pred_nifti.get_fdata(), dtype=torch.int8).unsqueeze(0)  # Add batch dimension

    # Load ground truth segmentation
    gt_path = os.path.join(gt_dir, patient_id, f"{patient_id}_seg.nii.gz")
    gt_nifti = nib.load(gt_path)
    gt_data = torch.tensor(gt_nifti.get_fdata(), dtype=torch.int8).unsqueeze(0)  # Add batch dimension

    # Compute Dice and HD95 for each patient
    dice_metric(y_pred=pred_data, y=gt_data)
    hd95_metric(y_pred=pred_data, y=gt_data)
    sensitivity_metric(y_pred=pred_data, y=gt_data)
    specificity_metric(y_pred=pred_data, y=gt_data)

# Aggregate results
dice_scores = dice_metric.aggregate().cpu().numpy()
hd95_scores = hd95_metric.aggregate().cpu().numpy()
sensitivity_scores = sensitivity_metric.aggregate().cpu().numpy()
specificity_scores = specificity_metric.aggregate().cpu().numpy()

# Print results
print(f"Average Dice Scores: WT - {dice_scores[0]:.4f}, TC - {dice_scores[1]:.4f}, ET - {dice_scores[2]:.4f}")
print(f"Average HD95 Scores: WT - {hd95_scores[0]:.4f}, TC - {hd95_scores[1]:.4f}, ET - {hd95_scores[2]:.4f}")
print(f"Average Sensitivity: WT - {sensitivity_scores[0]:.4f}, TC - {sensitivity_scores[1]:.4f}, ET - {sensitivity_scores[2]:.4f}")
print(f"Average Specificity: WT - {specificity_scores[0]:.4f}, TC - {specificity_scores[1]:.4f}, ET - {specificity_scores[2]:.4f}")