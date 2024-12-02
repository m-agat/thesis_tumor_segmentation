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
    compute_metrics_with_monai,
    calculate_composite_score,
)
from monai.metrics import ConfusionMatrixMetric

# Load model weights
class_weights = pd.read_csv("/home/magata/results/metrics/model_performance_summary.csv", index_col=0).to_dict(orient="index")
normalized_class_weights = {}
for tissue_index in [0, 1, 2, 4]:  
    tissue_key = f"Composite_Score_{tissue_index}"
    tissue_weights = [class_weights[model_name][tissue_key] for model_name in class_weights]
    total_weight = sum(tissue_weights)
    normalized_class_weights[tissue_index] = {
        model_name: class_weights[model_name][tissue_key] / total_weight
        for model_name in class_weights
    }

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
    vnet: "VNet"
}

class_labels = [0, 1, 2, 4]
output_channel_to_class_label = {0: 1, 1: 2, 2: 4}
weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
patient_scores = []
total_patients = len(config.test_loader_patient)
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False
)
save_interval = 1
results_save_path = os.path.join(config.output_dir, f"patient_performance_scores_ensemble.csv")

with torch.no_grad():
    for idx, batch_data in enumerate(config.test_loader_patient):
        image = batch_data["image"].to(config.device)
        patient_path = batch_data["path"]
        ground_truth = batch_data["label"][0].cpu().numpy()


        unique_values = np.unique(ground_truth)
        print(f"Unique values in ground truth: {unique_values}")
        if np.array_equal(unique_values, [0, 1]):
            print("Ground truth is binary.")
            break
        else:
            print("Ground truth is not binary.")
            break
        
        print(f"Processing patient {idx + 1}/{len(config.test_loader_patient)}: {patient_path[0]}")

        # Inference for each model
        models_inferers = {
            swinunetr: swinunetr_inferer, 
            segresnet: segresnet_inferer,
            attunet: attunet_inferer, 
            vnet: vnet_inferer
        }

        # Get prediction from each model for a given patient
        individual_case_predictions = []
        for model, inferer in models_inferers.items():
            prob = torch.sigmoid(inferer(image))
            seg = prob[0].detach().cpu().numpy() # batch_size, tissue, H, W, D
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2 
            seg_out[seg[0] == 1] = 1 
            seg_out[seg[2] == 1] = 4 
            individual_case_predictions.append(seg_out)
        
        individual_case_predictions = np.array(individual_case_predictions, dtype=np.int32) # shape: [num_models, H, W, D]

        # Weighted majority voting ensemble 
        # Initialize an array to store weighted votes for all classes
        class_labels = [0, 1, 2, 4] 
        num_classes = len(class_labels)
        weighted_votes = np.zeros((num_classes,) + individual_case_predictions[0].shape, dtype=np.float32)  # Shape: (num_classes, H, W, D)

        for tissue_type in class_labels:
            # Map class labels [0, 1, 2, 4] to indices 0, 1, 2, 3 in the weighted_votes array
            class_vote_idx = class_labels.index(tissue_type)
            
            for model_idx, model_pred in enumerate(individual_case_predictions):
                model_name = list(model_name_map.values())[model_idx]
                weighted_votes[class_vote_idx] += (model_pred == tissue_type) * normalized_class_weights[tissue_type][model_name]

        # THE CLASS WITH THE HIGHEST WEIGHTED VOTE IS SELECTED FOR EACH VOXEL 
        # This will give us the index in class_labels (0 to 3), so we need to map it back to the original labels (0, 1, 2, 4)
        final_segmentation_indices = np.argmax(weighted_votes, axis=0)

        # Map the indices back to the original class labels (0, 1, 2, 4)
        final_segmentation = np.zeros_like(final_segmentation_indices, dtype=np.int32)
        for i, class_label in enumerate(class_labels):
            final_segmentation[final_segmentation_indices == i] = class_label

        # Save the ensemble segmentation as a nifti file
        original_image_path = os.path.join(config.test_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz")
        original_nifti = nib.load(original_image_path)
        affine = original_nifti.affine
        ensemble_nifti = nib.Nifti1Image(final_segmentation, affine)
        save_path = os.path.join(config.output_dir, f"{patient_path[0]}_ensemble_segmentation.nii.gz")
        nib.save(ensemble_nifti, save_path)
        print(f"Saved ensemble segmentation for patient {patient_path[0]} at {save_path}")

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
                composite_score = calculate_composite_score(dice_score, hd95, f1_score, weights)
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

if patient_scores:
    pd.DataFrame(patient_scores).to_csv(
        results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
    )
    print(f"Saved final results to {results_save_path}")

print("Processing complete. All metrics saved.")