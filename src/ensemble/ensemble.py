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
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_sensitivity,
    calculate_composite_score,
    compute_f1_score,
    compute_specificity
)

base_path = "./outputs"
os.makedirs(base_path, exist_ok=True)
print(f"Output directory created: {base_path}")

# Load model weights
class_weights = pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/model_weights.csv", index_col=0).to_dict(orient="index")
normalized_class_weights = {}
for tissue in ["1", "2", "4"]:
    tissue_weights = [class_weights[model_name][tissue] for model_name in class_weights]
    total_weight = sum(tissue_weights)
    normalized_class_weights[int(tissue)] = {
        model_name: class_weights[model_name][tissue] / total_weight
        for model_name in class_weights
    }

print(class_weights)

# Define model paths
model_paths = {
    "swinunetr": "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/swinunetr_model.pt",
    "segresnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/segresnet_model.pt",
    "attunet": "/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/attunet_model.pt",
    "vnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/vnet_model.pt"
}

# Define function to load models
def load_model(model_class, checkpoint_path, device):
    model = model_class
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, partial(sliding_window_inference, roi_size=config.roi, sw_batch_size=1, predictor=model, overlap=0.6)

# Load models and inference configurations
swinunetr, swinunetr_inferer = load_model(models.swinunetr_model, model_paths["swinunetr"], config.device)
segresnet, segresnet_inferer = load_model(models.segresnet_model, model_paths["segresnet"], config.device)
attunet, attunet_inferer = load_model(models.attunet_model, model_paths["attunet"], config.device)
vnet, vnet_inferer = load_model(models.vnet_model, model_paths["vnet"], config.device)

model_name_map = {
    swinunetr: "SwinUNetr",
    segresnet: "SegResNet",
    attunet: "AttentionUNet",
    vnet: "VNet"
}

# Ensemble inference
patient_scores = []
tissue_averages = {
    1: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
    2: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
    4: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
}
weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
tissue_channel_map = {1: 0, 2: 1, 4: 2}

with torch.no_grad():
    for idx, batch_data in enumerate(config.test_loader_subset):
        image = batch_data["image"]
        patient_path = batch_data["path"]
        
        print(f"Processing patient {idx + 1}/{len(config.test_loader_subset)}: {patient_path[0]}")

        # Inference for each model
        prob_maps = {1: [], 2: [], 4: []}
        models_inferers = {
            swinunetr: swinunetr_inferer, segresnet: segresnet_inferer,
            attunet: attunet_inferer, vnet: vnet_inferer
        }

        for model, inferer in models_inferers.items():
            prob = torch.sigmoid(inferer(image)).cpu().numpy()[0]
            model_name = model_name_map[model]

            for tissue in [1, 2, 4]:
                tissue_index = tissue_channel_map[tissue]
                binary_prediction = (prob[tissue_index] > 0.5).astype(np.int8)
                weighted_prediction = binary_prediction * class_weights[model_name][str(tissue)]
                prob_maps[tissue].append(weighted_prediction)

        ensemble_seg = np.zeros_like(prob_maps[1][0])
        for tissue, weighted_votes in prob_maps.items():
            total_votes = np.sum(weighted_votes, axis=0)
            tissue_mask = (total_votes > 0.5).astype(np.int8)
            ensemble_seg[tissue_mask == 1] = tissue

        ground_truth = batch_data["label"][0].cpu().numpy()

        # Save the ensemble segmentation as a nifti file
        original_image_path = os.path.join(config.test_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz")
        original_nifti = nib.load(original_image_path)
        affine = original_nifti.affine
        ensemble_nifti = nib.Nifti1Image(ensemble_seg, affine)
        save_path = os.path.join(base_path, f"{patient_path[0]}_ensemble_segmentation.nii.gz")
        nib.save(ensemble_nifti, save_path)
        print(f"Saved ensemble segmentation for patient {patient_path[0]} at {save_path}")

        # Compute metrics
        patient_metrics = {"Patient": patient_path[0]}
        for tissue_type in [1, 2, 4]:
            dice_score = compute_dice_score_per_tissue(ensemble_seg, ground_truth, tissue_type)
            hd95 = compute_hd95(ensemble_seg, ground_truth, tissue_type)
            sensitivity = compute_sensitivity(ensemble_seg, ground_truth, tissue_type)
            specificity = compute_specificity(ensemble_seg, ground_truth, tissue_type)
            f1_score = compute_f1_score(ensemble_seg, ground_truth, tissue_type)
            composite_score = calculate_composite_score(dice_score, hd95, f1_score, weights)

            # Store metrics
            patient_metrics[f"Dice_{tissue_type}"] = dice_score
            patient_metrics[f"HD95_{tissue_type}"] = hd95
            patient_metrics[f"Sensitivity_{tissue_type}"] = sensitivity
            patient_metrics[f"Specificity_{tissue_type}"] = specificity
            patient_metrics[f"F1_{tissue_type}"] = f1_score
            patient_metrics[f"Composite_Score_{tissue_type}"] = composite_score

            # Update tissue averages
            tissue_averages[tissue_type]["Dice"].append(dice_score)
            tissue_averages[tissue_type]["HD95"].append(hd95)
            tissue_averages[tissue_type]["Sensitivity"].append(sensitivity)
            tissue_averages[tissue_type]["Specificity"].append(specificity)
            tissue_averages[tissue_type]["F1"].append(f1_score)
            tissue_averages[tissue_type]["Composite_Score"].append(composite_score)

        patient_scores.append(patient_metrics)
        print(patient_metrics)

# Save results
df_patient_scores = pd.DataFrame(patient_scores)
df_patient_scores.to_csv(os.path.join(base_path, "patient_scores_ensemble.csv"), index=False)

def mean_excluding_inf(values):
    finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
    return np.mean(finite_values) if finite_values else 0.0 

avg_scores = {
    "Tissue": ["1", "2", "4"],
    "Average Dice": [
        mean_excluding_inf(tissue_averages[1]["Dice"]),
        mean_excluding_inf(tissue_averages[2]["Dice"]),
        mean_excluding_inf(tissue_averages[4]["Dice"]),
    ],
    "Average HD95": [
        mean_excluding_inf(tissue_averages[1]["HD95"]),
        mean_excluding_inf(tissue_averages[2]["HD95"]),
        mean_excluding_inf(tissue_averages[4]["HD95"]),
    ],
    "Average Sensitivity": [
        mean_excluding_inf(tissue_averages[1]["Sensitivity"]),
        mean_excluding_inf(tissue_averages[2]["Sensitivity"]),
        mean_excluding_inf(tissue_averages[4]["Sensitivity"]),
    ],
    "Average Specificity": [
        mean_excluding_inf(tissue_averages[1]["Specificity"]),
        mean_excluding_inf(tissue_averages[2]["Specificity"]),
        mean_excluding_inf(tissue_averages[4]["Specificity"]),
    ],
    "Average F1": [
        mean_excluding_inf(tissue_averages[1]["F1"]),
        mean_excluding_inf(tissue_averages[2]["F1"]),
        mean_excluding_inf(tissue_averages[4]["F1"]),
    ],
    "Average Composite Score": [
        mean_excluding_inf(tissue_averages[1]["Composite_Score"]),
        mean_excluding_inf(tissue_averages[2]["Composite_Score"]),
        mean_excluding_inf(tissue_averages[4]["Composite_Score"]),
    ],
}
df_avg_scores = pd.DataFrame(avg_scores)
df_avg_scores.to_csv(os.path.join(base_path, "average_scores_ensemble.csv"), index=False)

print("Saved individual and average metrics.")
