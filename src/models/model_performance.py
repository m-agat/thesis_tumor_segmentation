import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import time
import pandas as pd
import json
import re

# Add custom modules
sys.path.append("../")
import models as models
import config.config as config
from utils.utils import AverageMeter
from dataset import dataloaders
from ensemble.ensemble_utils import compute_metrics, extract_patient_id, save_metrics, save_nifti, load_model


def performance_estimation(
    data_loader, model_inferer, patient_id=None, output_dir=config.output_dir
):
    """
    Perform segmentation using an ensemble of multiple models with simple averaging.

    Parameters:
    - patient_id: ID of the patient whose scan is being segmented.
    - data_loader: Dataloader for the test set.
    - models_dict: Dictionary containing trained models and their inferers.
    - output_dir: Directory where segmentations will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    if patient_id is not None:
        # Get a subset of test loader with the specific patient
        test_data_loader = config.find_patient_by_id(patient_id, data_loader)
    else:
        # Get full test data loader
        test_data_loader = data_loader

    patient_metrics = []

    with torch.no_grad():
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            patient_id = extract_patient_id(reference_image_path)
            gt = batch_data["label"].to(
                config.device
            )  # shape: (batch_size, 240, 240, 155)

            print(
                f"\nProcessing patient: {patient_id}\n",
            )

            # Collect logits from each model
            logits = model_inferer(image).squeeze(
                0
            )  # Remove batch dim -> (num_classes, H, W, D)

            # Apply softmax and convert to segmentation map
            seg = (
                torch.nn.functional.softmax(logits, dim=0).argmax(dim=0).unsqueeze(0)
            )  # Shape: (H, W, D)

            pred_one_hot = [(seg == i).float() for i in range(0, 4)]
            if gt.shape[1] == 4:
                # Ground truth is already one-hot encoded (assume channel 0 is background).
                # Extract channels 1,2,3 and permute to have shape [3, 1, H, W, D].
                gt_one_hot = gt.permute(1, 0, 2, 3, 4)
            else:
                # Ground truth is not one-hot encoded, so create one-hot encoding.
                gt_one_hot = [(gt == i).float() for i in range(1, 4)]
                gt_one_hot = torch.stack(gt_one_hot)


            dice, hd95, sensitivity, specificity = compute_metrics(
                pred_one_hot, gt_one_hot
            )
            print(
                f"Dice BG: {dice[0].item():.4f}, Dice NCR: {dice[1].item():.4f}, Dice ED: {dice[2].item():.4f}, Dice ET: {dice[3].item():.4f}\n"
                f"HD95 BG: {hd95[0].item():.2f}, HD95 NCR: {hd95[1].item():.2f}, HD95 ED: {hd95[2].item():.2f}, HD95 ET: {hd95[3].item():.2f}\n"
                f"Sensitivity BG: {sensitivity[0].item():.4f}, NCR: {sensitivity[1].item():.4f}, ED: {sensitivity[2].item():.4f}, ET: {sensitivity[3].item():.4f}\n"
                f"Specificity BG: {specificity[0].item():.4f}, NCR: {specificity[1].item():.4f}, ED: {specificity[2].item():.4f}, ET: {specificity[3].item():.4f}\n"
            )

            patient_metrics.append(
                {
                    "patient_id": patient_id,
                    "Dice BG": dice[0].item(),
                    "Dice NCR": dice[1].item(),
                    "Dice ED": dice[2].item(),
                    "Dice ET": dice[3].item(),
                    "Dice overall": float(np.nanmean(dice)),
                    "HD95 BG": hd95[0].item(),
                    "HD95 NCR": hd95[1].item(),
                    "HD95 ED": hd95[2].item(),
                    "HD95 ET": hd95[3].item(),
                    "HD95 overall": float(np.nanmean(hd95)),
                    "Sensitivity BG": sensitivity[0].item(),
                    "Sensitivity NCR": sensitivity[1].item(),
                    "Sensitivity ED": sensitivity[2].item(),
                    "Sensitivity ET": sensitivity[3].item(),
                    "Sensitivity overall": float(np.nanmean(sensitivity)),
                    "Specificity BG": specificity[0].item(),
                    "Specificity NCR": specificity[1].item(),
                    "Specificity ED": specificity[2].item(),
                    "Specificity ET": specificity[3].item(),
                    "Specificity overall": float(np.nanmean(specificity)),
                }
            )

            seg = seg.squeeze(0)  # remove batch dimension

            # Save segmentation
            # output_path = os.path.join(output_dir, f"segmentation_{patient_id}.nii.gz")
            # save_nifti(seg, reference_image_path, output_path)

            torch.cuda.empty_cache()

    save_metrics(patient_metrics, output_dir)


if __name__ == "__main__":
    patient_id = "01548"
    _, val_loader = dataloaders.get_loaders(
        batch_size=config.batch_size,
        json_path=config.json_path,
        basedir=config.root_dir,
        fold=None,
        roi=config.roi,
        use_final_split=True,
    )
    model, model_inferer = load_model(
        models.swinunetr_model, config.model_paths["swinunetr"], config.device
    )
    performance_estimation(val_loader, model_inferer, patient_id=patient_id)
