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
from test_time_dropout import ttd_variance, ttd_entropy
from test_time_augmentation import tta_variance, tta_entropy
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, partial(sliding_window_inference, roi_size=config.roi, sw_batch_size=config.sw_batch_size, predictor=model, overlap=config.infer_overlap)

# Load models and inference configurations
swinunetr, swinunetr_inferer = load_model(models.swinunetr_model, config.model_paths["swinunetr"], config.device)
segresnet, segresnet_inferer = load_model(models.segresnet_model, config.model_paths["segresnet"], config.device)
attunet, attunet_inferer = load_model(models.attunet_model, config.model_paths["attunet"], config.device)
vnet, vnet_inferer = load_model(models.vnet_model, config.model_paths["vnet"], config.device)

model_name_map = {
    swinunetr: "SwinUNetr",
    segresnet: "SegResNet",
    attunet: "AttentionUNet",
}

# Load model weights
class_weights = pd.read_csv("/home/magata/results/metrics/model_performance_summary.csv", index_col=0).to_dict(orient="index")
normalized_class_weights = {}
for tissue_index in [1, 2, 4]:  
    tissue_key = f"Dice_{tissue_index}"
    tissue_weights = [class_weights[model_name][tissue_key] for model_name in class_weights]
    total_weight = sum(tissue_weights)
    normalized_class_weights[tissue_index] = {
        model_name: class_weights[model_name][tissue_key] / total_weight
        for model_name in class_weights
    }

# MIscellanuous
class_labels = [1, 2, 4]
composite_score_weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False
)
patient_scores = []
results_save_path = os.path.join(config.output_dir, f"ensemble_performance_no_vnet.csv")
save_interval = 1
for idx, batch_data in enumerate(config.test_loader):
    # Get data and its path
    image = batch_data["image"].to(config.device)
    patient_path = batch_data["path"]

    print(f"Processing patient {idx + 1}/{len(config.test_loader)}: {patient_path[0]}")
    
    # Get ground truth and the affine matrix to save uncertainties as nifti later
    ground_truth = batch_data["label"][0].cpu().numpy()
    original_image_path = os.path.join(config.test_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz")
    original_nifti = nib.load(original_image_path)
    affine = original_nifti.affine

    models_inferers = {
            swinunetr: swinunetr_inferer, 
            segresnet: segresnet_inferer,
            attunet: attunet_inferer, 
        }
    
    ensemble_uncertainties = []  # To store uncertainty maps from all models
    ensemble_predictions = []    # To store segmentation maps from all models
    for model, model_inferer in models_inferers.items():
        mean_pred_ttd, var_pred_ttd = ttd_variance(model, image, model_inferer, n_iterations=20)
        mean_pred_tta, var_pred_tta = tta_variance(model_inferer, image, config.device, n_iterations=20)

        entr_mean_pred_ttd, entropy_ttd = ttd_entropy(model, image, model_inferer, n_iterations=20)
        entr_mean_pred_tta, entropy_tta = tta_entropy(model_inferer, image, config.device, n_iterations=20)
        
        # Combine uncertainties
        weight_ttd = 0.25  # Weight for TTD var
        weight_tta = 0.25  # Weight for TTA var
        weight_ttd_entropy = 0.25  # Weight for TTD entropy
        weight_tta_entropy = 0.25  # Weight for TTA entropy
        combined_uncertainty_raw = (
            weight_ttd * var_pred_ttd[0] + 
            weight_tta * var_pred_tta[0] + 
            weight_ttd_entropy * np.repeat(entropy_ttd[np.newaxis, ...], repeats=3, axis=0) + 
            weight_tta_entropy * np.repeat(entropy_tta[np.newaxis, ...], repeats=3, axis=0)
        )
        combined_uncertainty = (combined_uncertainty_raw - np.min(combined_uncertainty_raw)) / (np.max(combined_uncertainty_raw) - np.min(combined_uncertainty_raw))
        print(f"Combined uncertainty shape: {combined_uncertainty.shape}")

        # Get the predicted classes
        combined_prediction_tensor = (
            0.5 * torch.sigmoid(torch.tensor(mean_pred_ttd, device=config.device)) +
            0.5 * torch.sigmoid(torch.tensor(mean_pred_tta, device=config.device))
        )
        combined_prediction_tensor = (combined_prediction_tensor + torch.tensor(entr_mean_pred_ttd, device=config.device) + torch.tensor(entr_mean_pred_tta, device=config.device)) / 3
        combined_prediction = combined_prediction_tensor.cpu().numpy()[0]
        print("Combined prediction shape: ", combined_prediction.shape)

        ensemble_predictions.append(combined_prediction)
        ensemble_uncertainties.append(combined_uncertainty)      

    ensemble_predictions = np.stack(ensemble_predictions, axis=0)  # Shape: (num_models, tissues, height, width, depth)
    ensemble_uncertainties = np.stack(ensemble_uncertainties, axis=0)  # Shape: (num_models, tissues, height, width, depth)

    # Weighted combination
    weights = 1 / (ensemble_uncertainties + 1e-6)  # Inverse uncertainty as weight
    weights /= np.sum(weights, axis=0, keepdims=True)  # Normalize weights
    weighted_probabilities = np.sum(weights * ensemble_predictions, axis=0)  # Shape: (3, height, width, depth)

    # Generate final segmentation
    final_segmentation = np.zeros(weighted_probabilities.shape[1:], dtype=np.int8)
    seg = (weighted_probabilities > 0.5).astype(np.int8)
    final_segmentation[seg[1] == 1] = 2  # ED (Whole Tumor = ED + NCR + ET)
    final_segmentation[seg[0] == 1] = 1  # NCR (Tumor Core = NCR + ET)
    final_segmentation[seg[2] == 1] = 4  # ET (Enhancing Tumor)
    
    final_segmentation_map = nib.Nifti1Image(final_segmentation, affine)
    ensemble_save_path = os.path.join(config.output_dir, f"{patient_path[0]}_ensemble_segmentation.nii.gz")
    nib.save(final_segmentation_map, ensemble_save_path)
    print(f"Saved ensemble segmentation for patient {patient_path[0]} at {ensemble_save_path}")

    # Compute ensemble uncertainty
    epistemic_uncertainty = np.var(ensemble_predictions, axis=0)
    aleatoric_uncertainty = np.mean(ensemble_uncertainties, axis=0)
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty

    # Create uncertainty map
    uncertainty_map = np.zeros(seg.shape[1:], dtype=np.float32)
    uncertainty_map[seg[1] == 1] = total_uncertainty[1][seg[1] == 1]  # ED (Whole Tumor)
    uncertainty_map[seg[0] == 1] = total_uncertainty[0][seg[0] == 1]  # NCR/NET (Tumor Core)
    uncertainty_map[seg[2] == 1] = total_uncertainty[2][seg[2] == 1]  # ET (Enhancing Tumor)

    # Save uncertainty map as NIfTI
    uncertainty_map_img = nib.Nifti1Image(uncertainty_map, affine)
    uncertainty_save_path = os.path.join(config.output_dir, f"{patient_path[0]}_ensemble_uncertainty_map.nii.gz")
    nib.save(uncertainty_map_img, uncertainty_save_path)
    print(f"Saved uncertainty map for patient {patient_path[0]} at {uncertainty_save_path}")

    patient_metrics = {"Patient": patient_path[0]}
    for tissue_type in class_labels:
        print(f"  Computing metrics for tissue type {tissue_type}")
        # Compute and store metrics
        try:
            dice_score = compute_dice_score_per_tissue(final_segmentation, ground_truth, tissue_type)
            print(f"    Dice score for tissue {tissue_type}: {dice_score}")
        except Exception as e:
            print(f"    Error computing Dice for tissue {tissue_type}: {e}")
            dice_score = np.nan
        try:
            hd95 = compute_hd95(final_segmentation, ground_truth, tissue_type)
            if np.isnan(hd95):
                print(f"    HD95 for tissue {tissue_type} not computable, setting to 0")
                hd95 = float('inf')
            print(f"    HD95 for tissue {tissue_type}: {hd95}")
        except Exception as e:
            print(f"    Error computing HD95 for tissue {tissue_type}: {e}")
            hd95 = np.nan
        try:
            sensitivity, specificity, f1_score = compute_metrics_with_monai(final_segmentation, ground_truth, tissue_type, confusion_metric)
            print(f"    Sensitivity for tissue {tissue_type}: {sensitivity}")
            print(f"    Specificity for tissue {tissue_type}: {specificity}")
            print(f"    F1 score for tissue {tissue_type}: {f1_score}")
            if np.isnan(sensitivity) or np.isnan(f1_score):
                print(f"    Sensitivity or F1 for tissue {tissue_type} is not computable, setting to 0")
                sensitivity, f1_score = 0.0, 0.0
        except Exception as e:
            print(f"    Error computing F1 Score for tissue {tissue_type}: {e}")
            sensitivity, specificity, f1_score = np.nan, np.nan, np.nan
        try:
            composite_score = calculate_composite_score(dice_score, hd95, f1_score, composite_score_weights)
            print(f"    Composite score for tissue {tissue_type}: {composite_score}")
        except Exception as e:
            print(f"    Error computing Composite Score for tissue {tissue_type}: {e}")
            composite_score = np.nan

        patient_metrics.update({
            f"Dice_{tissue_type}": dice_score,
            f"HD95_{tissue_type}": hd95,
            f"Sensitivity_{tissue_type}": sensitivity,
            f"Specificity_{tissue_type}": specificity,
            f"F1_{tissue_type}": f1_score,
            f"Composite_Score_{tissue_type}": composite_score,
        })

    patient_scores.append(patient_metrics)
    print(f"Metrics for patient {patient_path[0]}: {patient_metrics}")

    if idx % save_interval == 0:
        print(f"Saving intermediate results at patient index {idx}")
        pd.DataFrame(patient_scores).to_csv(
            results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
        )
        print(f"Saved intermediate results to {results_save_path}")
        patient_scores.clear()


if patient_scores:
    pd.DataFrame(patient_scores).to_csv(
        results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
    )
    print(f"Saved final results to {results_save_path}")

print("Processing complete. All metrics saved.")

        