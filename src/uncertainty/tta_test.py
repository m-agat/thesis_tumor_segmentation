import os
import sys

sys.path.append("../")
import nibabel as nib
import models.models as models
import torch
import config.config as config
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import pandas as pd
from test_time_augmentation import test_time_augmentation_inference
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_metrics_with_monai,
    calculate_composite_score,
)
from monai.metrics import ConfusionMatrixMetric


# Define function to load models
def load_model(model_class, checkpoint_path, device):
    model = model_class
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, partial(
        sliding_window_inference,
        roi_size=config.roi,
        sw_batch_size=config.sw_batch_size,
        predictor=model,
        overlap=config.infer_overlap,
    )


def normalize(data, min_val=0, max_val=1):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (
        max_val - min_val
    ) + min_val


# Load models and inference configurations
swinunetr, swinunetr_inferer = load_model(
    models.swinunetr_model, config.model_paths["swinunetr"], config.device
)
segresnet, segresnet_inferer = load_model(
    models.segresnet_model, config.model_paths["segresnet"], config.device
)
attunet, attunet_inferer = load_model(
    models.attunet_model, config.model_paths["attunet"], config.device
)
vnet, vnet_inferer = load_model(
    models.vnet_model, config.model_paths["vnet"], config.device
)

model_name_map = {
    swinunetr: "SwinUNetr",
    segresnet: "SegResNet",
    attunet: "AttentionUNet",
    vnet: "VNet",
}
# MIscellanuous
class_labels = [0, 1, 2, 4]
weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False,
)
patient_scores = []
results_save_path = os.path.join(config.output_dir, f"tta_ensemble_performance.csv")
save_interval = 1
for idx, batch_data in enumerate(config.test_loader_patient):
    # Get data and its path
    image = batch_data["image"].to(config.device)
    patient_path = batch_data["path"]

    print(
        f"Processing patient {idx + 1}/{len(config.test_loader_patient)}: {patient_path[0]}"
    )

    # Get ground truth and the affine matrix to save uncertainties as nifti later
    ground_truth = batch_data["label"][0].cpu().numpy()
    original_image_path = os.path.join(
        config.test_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz"
    )
    original_nifti = nib.load(original_image_path)
    affine = original_nifti.affine

    models_inferers = {
        swinunetr: swinunetr_inferer,
        segresnet: segresnet_inferer,
        attunet: attunet_inferer,
        vnet: vnet_inferer,
    }

    ensemble_uncertainties = []  # To store uncertainty maps from all models
    ensemble_predictions = []  # To store segmentation maps from all models
    for model, model_inferer in models_inferers.items():
        mean_pred, var_pred = test_time_augmentation_inference(
            model_inferer, image, config.device, n_iterations=3
        )

        # Get the predicted classes
        mean_pred_tensor = torch.tensor(mean_pred, device=config.device)
        mean_pred_tensor = torch.sigmoid(mean_pred_tensor)
        mean_pred = mean_pred_tensor.cpu().numpy()[0]
        predicted_labels = (mean_pred > 0.5).astype(np.int8)

        # Get predicted labels
        seg_out = np.zeros(
            (
                predicted_labels.shape[1],
                predicted_labels.shape[2],
                predicted_labels.shape[3],
            )
        )
        seg_out[predicted_labels[1] == 1] = 2  # ED (Whole Tumor = ED + NCR + ET)
        seg_out[predicted_labels[0] == 1] = 1  # NCR/NET (Tumor Core = NCR + ET)
        seg_out[predicted_labels[2] == 1] = 4  # ET (Enhancing Tumor)

        # Assign uncertainties to predicted cells
        voxel_uncertainties = var_pred[0]  # remove batch dimension
        cell_type_uncertainty = np.zeros(
            (
                predicted_labels.shape[1],
                predicted_labels.shape[2],
                predicted_labels.shape[3],
            )
        )
        cell_type_uncertainty[predicted_labels[1] == 1] = voxel_uncertainties[1][
            predicted_labels[1] == 1
        ]  # ED (Whole Tumor)
        cell_type_uncertainty[predicted_labels[0] == 1] = voxel_uncertainties[0][
            predicted_labels[0] == 1
        ]  # NCR/NET (Tumor Core)
        cell_type_uncertainty[predicted_labels[2] == 1] = voxel_uncertainties[2][
            predicted_labels[2] == 1
        ]  # ET (Enhancing Tumor)

        # uncertainty_map = nib.Nifti1Image(cell_type_uncertainty, affine)
        # print(f"Shape of uncertainty map: {uncertainty_map.shape}")
        # save_path = os.path.join(config.output_dir, f"{patient_path[0]}_{model_name_map[model]}_tta_uncertainty_map.nii.gz")
        # nib.save(uncertainty_map, save_path)
        # print(f"Saved ensemble segmentation for patient {patient_path[0]} at {save_path}")

        ensemble_predictions.append(seg_out)
        ensemble_uncertainties.append(cell_type_uncertainty)

    ensemble_predictions = np.stack(
        ensemble_predictions, axis=0
    )  # Shape: (num_models, height, width, depth)
    ensemble_uncertainties = np.stack(
        ensemble_uncertainties, axis=0
    )  # Shape: (num_models, height, width, depth)

    # # Create final ensemble prediction
    # # For each voxel, choose the label from the model with the lowest uncertainty
    final_segmentation = np.zeros_like(ensemble_predictions[0], dtype=np.int8)
    for x in range(ensemble_predictions.shape[1]):
        for y in range(ensemble_predictions.shape[2]):
            for z in range(ensemble_predictions.shape[3]):
                # Get uncertainties and labels for this voxel from all models
                voxel_uncertainties = ensemble_uncertainties[:, x, y, z]
                voxel_labels = ensemble_predictions[:, x, y, z]

                # Find the model with the lowest uncertainty
                best_model_idx = np.argmin(voxel_uncertainties)
                final_segmentation[x, y, z] = voxel_labels[best_model_idx]

    final_segmentation_map = nib.Nifti1Image(final_segmentation, affine)
    ensemble_save_path = os.path.join(
        config.output_dir, f"{patient_path[0]}_tta_ensemble_segmentation.nii.gz"
    )
    nib.save(final_segmentation_map, ensemble_save_path)
    print(
        f"Saved ensemble segmentation for patient {patient_path[0]} at {ensemble_save_path}"
    )

    patient_metrics = {"Patient": patient_path[0]}
    for tissue_type in class_labels:
        print(f"  Computing metrics for tissue type {tissue_type}")
        # Compute and store metrics
        try:
            dice_score = compute_dice_score_per_tissue(
                final_segmentation, ground_truth, tissue_type
            )
            print(f"    Dice score for tissue {tissue_type}: {dice_score}")
        except Exception as e:
            print(f"    Error computing Dice for tissue {tissue_type}: {e}")
            dice_score = np.nan
        try:
            hd95 = compute_hd95(final_segmentation, ground_truth, tissue_type)
            if np.isnan(hd95):
                print(f"    HD95 for tissue {tissue_type} not computable, setting to 0")
                hd95 = float("inf")
            print(f"    HD95 for tissue {tissue_type}: {hd95}")
        except Exception as e:
            print(f"    Error computing HD95 for tissue {tissue_type}: {e}")
            hd95 = np.nan
        try:
            sensitivity, specificity, f1_score = compute_metrics_with_monai(
                final_segmentation, ground_truth, tissue_type, confusion_metric
            )
            print(f"    Sensitivity for tissue {tissue_type}: {sensitivity}")
            print(f"    Specificity for tissue {tissue_type}: {specificity}")
            print(f"    F1 score for tissue {tissue_type}: {f1_score}")
            if np.isnan(sensitivity) or np.isnan(f1_score):
                print(
                    f"    Sensitivity or F1 for tissue {tissue_type} is not computable, setting to 0"
                )
                sensitivity, f1_score = 0.0, 0.0
        except Exception as e:
            print(f"    Error computing F1 Score for tissue {tissue_type}: {e}")
            sensitivity, specificity, f1_score = np.nan, np.nan, np.nan
        try:
            composite_score = calculate_composite_score(
                dice_score, hd95, f1_score, weights
            )
            print(f"    Composite score for tissue {tissue_type}: {composite_score}")
        except Exception as e:
            print(f"    Error computing Composite Score for tissue {tissue_type}: {e}")
            composite_score = np.nan

        patient_metrics.update(
            {
                f"Dice_{tissue_type}": dice_score,
                f"HD95_{tissue_type}": hd95,
                f"Sensitivity_{tissue_type}": sensitivity,
                f"Specificity_{tissue_type}": specificity,
                f"F1_{tissue_type}": f1_score,
                f"Composite_Score_{tissue_type}": composite_score,
            }
        )

    patient_scores.append(patient_metrics)
    print(f"Metrics for patient {patient_path[0]}: {patient_metrics}")
