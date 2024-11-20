import os
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import gc
import config.config as config
import models.models as models
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import pandas as pd
import torch.nn as nn
from monai.losses import DiceLoss
from mri_feature_extraction import extract_regional_features
from monai.metrics import DiceMetric
import pickle as pkl 
import nibabel as nib 
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_metrics_with_monai,
    calculate_composite_score,
)
from monai.metrics import ConfusionMatrixMetric

# Set base output directory
base_path = config.output_dir
os.makedirs(base_path, exist_ok=True)
print(f"Output directory: {base_path}")

# Load model function
def load_all_models(): 
    model_map = {
        "swinunetr": models.swinunetr_model,
        "segresnet": models.segresnet_model,
        # "vnet": models.vnet_model,
        # "attunet": models.attunet_model,
    }
    
    loaded_models = {}
    for model_name, model_class in model_map.items():
        model = model_class.to(config.device).eval()
        checkpoint = torch.load(config.model_paths[model_name], map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        loaded_models[model_name] = model
    
    return loaded_models

def align_and_extract_features(model_predictions, image, feature_extraction_function=extract_regional_features):
    """
    Extract features for the aligned region based on the intersection or union of segmentations from both models.
    Args:
        model_predictions_swinunetr: Segmentation output from SwinUNETR.
        model_predictions_segresnet: Segmentation output from SegResNet.
        image: The original MRI image.
        region_name: The region to extract features from (e.g., "NCR").
        threshold: Threshold for segmentation (usually 0.5 for binary segmentation).
        feature_extraction_function: Function to extract features from the region.
        
    Returns:
        Aligned features for the region.
    """
    seg_swinunetr = model_predictions[0]
    seg_segresnet = model_predictions[1]

    region_mask_swinunetr_NCR = (seg_swinunetr == 1)  
    region_mask_segresnet_NCR = (seg_segresnet == 1)  

    region_mask_swinunetr_ED = (seg_swinunetr == 2)  
    region_mask_segresnet_ED = (seg_segresnet == 2)  

    region_mask_swinunetr_ET = (seg_swinunetr == 4)  
    region_mask_segresnet_ET = (seg_segresnet == 4)  

    masks = {"NCR": np.logical_or(region_mask_swinunetr_NCR, region_mask_segresnet_NCR),
             "ED": np.logical_or(region_mask_swinunetr_ED, region_mask_segresnet_ED),
             "ET": np.logical_or(region_mask_swinunetr_ET, region_mask_segresnet_ET)}

    # intersection_mask = np.logical_and(region_mask_swinunetr, region_mask_segresnet)  # Intersection

    features = feature_extraction_function(image[0], masks)
    return features

models_dict = load_all_models()

dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

selected_features_NCR = pd.read_csv("./outputs/combined_selected_features_NCR.csv")['Feature'].tolist()
selected_features_ED = pd.read_csv("./outputs/combined_selected_features_ED.csv")['Feature'].tolist()
selected_features_ET = pd.read_csv("./outputs/combined_selected_features_ET.csv")['Feature'].tolist()

class_labels = [0, 1, 2, 4]  
tissue_types = {1: "NCR", 2: "ED", 4: "ET"} 
num_classes = len(class_labels)
threshold = 0.5
n_epochs = 1

weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
patient_scores = []
total_patients = len(config.val_loader)
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False
)

for epoch in range(n_epochs):
    total_loss = 0  
    num_batches = len(config.test_loader_subset)

    for idx, batch_data in enumerate(config.test_loader_subset): 
        image = batch_data["image"].to(config.device)  
        ground_truth = batch_data["label"][0].to(config.device)
        patient_path = batch_data["path"]

        print(f"Processing patient {idx + 1}/{len(config.test_loader_subset)}: {patient_path[0]}")
                
        # Prepare model predictions and features list
        model_predictions = []
        with torch.no_grad():  
            for model_name, model in models_dict.items():
                # Get model prediction
                model_inferer = partial(
                    sliding_window_inference,
                    roi_size=config.roi,
                    sw_batch_size=config.sw_batch_size,
                    predictor=model,
                    overlap=config.infer_overlap,
                )
                
                prob = torch.sigmoid(model_inferer(image)).to(config.device)
                seg = (prob[0].detach().cpu().numpy() > threshold).astype(np.int8)
                seg_out = np.zeros(
                    (seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8
                )
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 4
                model_predictions.append(seg_out)

        mri_features = align_and_extract_features(model_predictions, image)

        NCR_features = [mri_features[feature] for feature in selected_features_NCR]
        ED_features = [mri_features[feature] for feature in selected_features_ED]
        ET_features = [mri_features[feature] for feature in selected_features_ET]

        NCR_model_path = f"./outputs/meta_learner_NCR.pkl" 
        ED_model_path = f"./outputs/meta_learner_ED.pkl"
        ET_model_path = f"./outputs/meta_learner_ET.pkl"

        with open(NCR_model_path, 'rb') as f:
            NCR_model = pkl.load(f)

        with open(ED_model_path, 'rb') as f:
            ED_model = pkl.load(f)

        with open(ET_model_path, 'rb') as f:
            ET_model = pkl.load(f)

        predicted_weights = {
            'NCR': {},
            'ED': {},
            'ET': {}
        }

        NCR_features = [feature if isinstance(feature, np.ndarray) else np.zeros_like(NCR_features[0]) for feature in NCR_features]
        NCR_features_reshaped = np.array(NCR_features).reshape(1, -1) 
        NCR_features_df = pd.DataFrame(NCR_features_reshaped, columns=selected_features_NCR)
        NCR_pred = NCR_model.predict(NCR_features_df)
        predicted_weights['NCR']['swinunetr'] = NCR_pred[:, 0]  
        predicted_weights['NCR']['segresnet'] = NCR_pred[:, 1] 
        print(f"NCR pred for swinunetr: {NCR_pred[:, 0]} and for segresnet: {NCR_pred[:, 1]}")

        ED_features = [feature if isinstance(feature, np.ndarray) else np.zeros_like(ED_features[0]) for feature in ED_features]
        ED_features_reshaped = np.array(ED_features).reshape(1, -1) 
        ED_features_df = pd.DataFrame(ED_features_reshaped, columns=selected_features_ED)
        ED_pred = ED_model.predict(ED_features_df)
        predicted_weights['ED']['swinunetr'] = ED_pred[:, 0]  
        predicted_weights['ED']['segresnet'] = ED_pred[:, 1] 
        print(f"ED pred for swinunetr: {ED_pred[:, 0]} and for segresnet: {ED_pred[:, 1]}")

        ET_features = [feature if isinstance(feature, np.ndarray) else np.zeros_like(ET_features[0]) for feature in ET_features]
        ET_features_reshaped = np.array(ET_features).reshape(1, -1) 
        ET_features_df = pd.DataFrame(ET_features_reshaped, columns=selected_features_ET)
        ET_pred = ET_model.predict(ET_features_df)
        predicted_weights['ET']['swinunetr'] = ET_pred[:, 0]  
        predicted_weights['ET']['segresnet'] = ET_pred[:, 1] 
        print(f"ET pred for swinunetr: {ET_pred[:, 0]} and for segresnet: {ET_pred[:, 1]}")

        model_predictions = np.array(model_predictions)
        weighted_votes = np.zeros((len(class_labels),) + model_predictions[0].shape, dtype=np.float32) 
        for class_idx in class_labels[1:]: 
            class_vote_idx = class_labels.index(class_idx)
            for model_idx, model_seg in enumerate(model_predictions):
                tissue_weight = predicted_weights[tissue_types[class_idx]][model_name]
                weighted_votes[class_vote_idx] += (model_seg == class_idx) * tissue_weight

        final_segmentation_indices = np.argmax(weighted_votes, axis=0)

        # Map the indices back to the original class labels (0, 1, 2, 4)
        final_segmentation = np.zeros_like(final_segmentation_indices, dtype=np.int32)
        for i, class_label in enumerate(class_labels):
            final_segmentation[final_segmentation_indices == i] = class_label

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
        
            # Calculate Dice, HD95, and other metrics for SwinUNETR, SegResNet, and Ensemble segmentation
            try:
                # SwinUNETR Segmentation Metrics
                swinunetr_dice = compute_dice_score_per_tissue(model_predictions[0], ground_truth, tissue_type)
                print(f"    Dice score for SwinUNETR tissue {tissue_type}: {swinunetr_dice}")
            except Exception as e:
                print(f"    Error computing Dice for SwinUNETR tissue {tissue_type}: {e}")
                swinunetr_dice = np.nan
                
            try:
                swinunetr_hd95 = compute_hd95(model_predictions[0], ground_truth, tissue_type)
                if np.isnan(swinunetr_hd95):
                    print(f"    HD95 for SwinUNETR tissue {tissue_type} not computable, setting to 0")
                    swinunetr_hd95 = float('inf')
                print(f"    HD95 for SwinUNETR tissue {tissue_type}: {swinunetr_hd95}")
            except Exception as e:
                print(f"    Error computing HD95 for SwinUNETR tissue {tissue_type}: {e}")
                swinunetr_hd95 = np.nan
                
            try:
                swinunetr_f1 = compute_metrics_with_monai(model_predictions[0], ground_truth, tissue_type, confusion_metric)[2]
                print(f"    F1 score for SwinUNETR tissue {tissue_type}: {swinunetr_f1}")
            except Exception as e:
                print(f"    Error computing F1 Score for SwinUNETR tissue {tissue_type}: {e}")
                swinunetr_f1 = np.nan
            
            # SegResNet Segmentation Metrics
            try:
                segresnet_dice = compute_dice_score_per_tissue(model_predictions[1], ground_truth, tissue_type)
                print(f"    Dice score for SegResNet tissue {tissue_type}: {segresnet_dice}")
            except Exception as e:
                print(f"    Error computing Dice for SegResNet tissue {tissue_type}: {e}")
                segresnet_dice = np.nan
                
            try:
                segresnet_hd95 = compute_hd95(model_predictions[1], ground_truth, tissue_type)
                if np.isnan(segresnet_hd95):
                    print(f"    HD95 for SegResNet tissue {tissue_type} not computable, setting to 0")
                    segresnet_hd95 = float('inf')
                print(f"    HD95 for SegResNet tissue {tissue_type}: {segresnet_hd95}")
            except Exception as e:
                print(f"    Error computing HD95 for SegResNet tissue {tissue_type}: {e}")
                segresnet_hd95 = np.nan
                
            try:
                segresnet_f1 = compute_metrics_with_monai(model_predictions[1], ground_truth, tissue_type, confusion_metric)[2]
                print(f"    F1 score for SegResNet tissue {tissue_type}: {segresnet_f1}")
            except Exception as e:
                print(f"    Error computing F1 Score for SegResNet tissue {tissue_type}: {e}")
                segresnet_f1 = np.nan
            
            # Ensemble Segmentation Metrics
            try:
                ensemble_dice = compute_dice_score_per_tissue(final_segmentation, ground_truth, tissue_type)
                print(f"    Dice score for Ensemble tissue {tissue_type}: {ensemble_dice}")
            except Exception as e:
                print(f"    Error computing Dice for Ensemble tissue {tissue_type}: {e}")
                ensemble_dice = np.nan
                
            try:
                ensemble_hd95 = compute_hd95(final_segmentation, ground_truth, tissue_type)
                if np.isnan(ensemble_hd95):
                    print(f"    HD95 for Ensemble tissue {tissue_type} not computable, setting to 0")
                    ensemble_hd95 = float('inf')
                print(f"    HD95 for Ensemble tissue {tissue_type}: {ensemble_hd95}")
            except Exception as e:
                print(f"    Error computing HD95 for Ensemble tissue {tissue_type}: {e}")
                ensemble_hd95 = np.nan
                
            try:
                ensemble_f1 = compute_metrics_with_monai(final_segmentation, ground_truth, tissue_type, confusion_metric)[2]
                print(f"    F1 score for Ensemble tissue {tissue_type}: {ensemble_f1}")
            except Exception as e:
                print(f"    Error computing F1 Score for Ensemble tissue {tissue_type}: {e}")
                ensemble_f1 = np.nan
            
            # Composite Score Calculation
            try:
                swinunetr_composite = calculate_composite_score(swinunetr_dice, swinunetr_hd95, swinunetr_f1, weights)
                print(f"    Composite score for SwinUNETR tissue {tissue_type}: {swinunetr_composite}")
            except Exception as e:
                print(f"    Error computing Composite Score for SwinUNETR tissue {tissue_type}: {e}")
                swinunetr_composite = np.nan
                
            try:
                segresnet_composite = calculate_composite_score(segresnet_dice, segresnet_hd95, segresnet_f1, weights)
                print(f"    Composite score for SegResNet tissue {tissue_type}: {segresnet_composite}")
            except Exception as e:
                print(f"    Error computing Composite Score for SegResNet tissue {tissue_type}: {e}")
                segresnet_composite = np.nan
                
            try:
                ensemble_composite = calculate_composite_score(ensemble_dice, ensemble_hd95, ensemble_f1, weights)
                print(f"    Composite score for Ensemble tissue {tissue_type}: {ensemble_composite}")
            except Exception as e:
                print(f"    Error computing Composite Score for Ensemble tissue {tissue_type}: {e}")
                ensemble_composite = np.nan

            patient_metrics[f"SwinUNETR_{tissue_type}_Dice"] = swinunetr_dice
            patient_metrics[f"SwinUNETR_{tissue_type}_HD95"] = swinunetr_hd95
            patient_metrics[f"SwinUNETR_{tissue_type}_F1"] = swinunetr_f1
            patient_metrics[f"SwinUNETR_{tissue_type}_Composite"] = swinunetr_composite

            patient_metrics[f"SegResNet_{tissue_type}_Dice"] = segresnet_dice
            patient_metrics[f"SegResNet_{tissue_type}_HD95"] = segresnet_hd95
            patient_metrics[f"SegResNet_{tissue_type}_F1"] = segresnet_f1
            patient_metrics[f"SegResNet_{tissue_type}_Composite"] = segresnet_composite

            patient_metrics[f"Ensemble_{tissue_type}_Dice"] = ensemble_dice
            patient_metrics[f"Ensemble_{tissue_type}_HD95"] = ensemble_hd95
            patient_metrics[f"Ensemble_{tissue_type}_F1"] = ensemble_f1
            patient_metrics[f"Ensemble_{tissue_type}_Composite"] = ensemble_composite
        
        patient_scores.append(patient_metrics)


metrics_df = pd.DataFrame(patient_scores)
metrics_df.to_csv("./outputs/0_patient_performance_comparison.csv", index=False)
print("Saved performance comparison to './outputs/patient_performance_comparison.csv'.")

#     # Average loss over all batches
#     average_loss = total_loss / num_batches
#     print(f"\tEpoch {epoch + 1} average loss: {average_loss:.4f}")
#     torch.cuda.empty_cache() 


# print("Meta-learner training complete.")