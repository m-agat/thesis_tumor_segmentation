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
import math

# Add custom modules
sys.path.append("../")
import models.models as models
import config.config as config
from uncertainty.test_time_dropout import ttd_variance, minmax_uncertainties
from dataset import dataloaders

#####################
#### Load Models ####
#####################

def load_model(model_class, checkpoint_path, device):
    """
    Load a segmentation model from a checkpoint.
    """
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
        mode="gaussian",    # <--- ensures patch edges blend smoothly
        sigma_scale=0.125,  # <--- default smoothing parameter
        padding_mode="constant",
    )

def load_all_models():
    """
    Load all segmentation models into a dictionary.
    """
    return {
        "swinunetr": load_model(
            models.swinunetr_model, config.model_paths["swinunetr"], config.device
        ),
        "segresnet": load_model(
            models.segresnet_model, config.model_paths["segresnet"], config.device
        ),
        "attunet": load_model(
            models.attunet_model, config.model_paths["attunet"], config.device
        ),
    }

#####################
#### Load Weights ####
#####################

def load_weights(performance_weights_path):
    with open(performance_weights_path) as f:
        performance = json.load(f)
    return performance

##############################
#### Compute Weighted Scores ##
##############################

def compute_composite_scores(metrics, weights):
    """Compute weighted composite scores for a model.
    Now includes a score for the background (BG) class.
    """
    composite_scores = {}
    # Background composite score
    normalized_hd95_bg = 1 / (1 + metrics["HD95 BG"])
    composite_scores["BG"] = (
        weights["Dice"] * metrics["Dice BG"]
        + weights["HD95"] * normalized_hd95_bg
        + weights["Sensitivity"] * metrics["Sensitivity BG"]
        + weights["Specificity"] * metrics["Specificity BG"]
    )
    # Tumor regions composite scores
    for region in ["NCR", "ED", "ET"]:
        normalized_hd95 = 1 / (1 + metrics[f"HD95 {region}"])
        composite_scores[region] = (
            weights["Dice"] * metrics[f"Dice {region}"]
            + weights["HD95"] * normalized_hd95
            + weights["Sensitivity"] * metrics[f"Sensitivity {region}"]
            + weights["Specificity"] * metrics[f"Specificity {region}"]
        )
    return composite_scores

#############################
#### Save Segmentation ######
#############################

def save_segmentation_as_nifti(predicted_segmentation, reference_image_path, output_path):
    """
    Save the predicted segmentation as a NIfTI file.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()
    predicted_segmentation = predicted_segmentation.astype(np.uint8)
    ref_img = nib.load(reference_image_path)
    seg_img = nib.Nifti1Image(predicted_segmentation, affine=ref_img.affine, header=ref_img.header)
    nib.save(seg_img, output_path)
    print(f"Segmentation saved to {output_path}")

def save_probability_map_as_nifti(prob_map, ref_img, output_path):
    """
    Save a probability map (float32) as a NIfTI file.
    """
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
    prob_map = prob_map.astype(np.float32)
    prob_img = nib.Nifti1Image(prob_map, affine=ref_img.affine, header=ref_img.header)
    nib.save(prob_img, output_path)
    print(f"Probability map saved to {output_path}")

########################################
#### Gather performance metrics ########
########################################

def compute_metrics(pred, gt):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    """
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.NONE, get_not_nans=True)
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity"],
        reduction="none",
        compute_sample=False,
    )
    pred = [p.detach().clone() if hasattr(p, "detach") else p for p in pred]
    gt = [g.detach().clone() if hasattr(g, "detach") else g for g in gt]
    pred_stack = torch.stack(pred)
    gt_stack = torch.stack(gt)
    dice_metric(y_pred=pred, y=gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.cpu().numpy()
    for i, dice_score in enumerate(dice_scores):
        if not_nans[i] == 0:  # Tissue is absent in ground truth
            pred_empty = torch.sum(pred_stack[i]).item() == 0
            dice_scores[i] = 1.0 if pred_empty else 0.0
    hd95 = compute_hausdorff_distance(
        y_pred=pred_stack,
        y=gt_stack,
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
    )
    hd95 = hd95.squeeze(0).cpu().numpy()
    for i in range(len(hd95)):
        pred_empty = torch.sum(pred[i]).item() == 0
        gt_empty = not_nans[i] == 0
        if pred_empty and gt_empty:
            print(f"Region {i}: Both GT and Prediction are empty. Setting HD95 to 0.")
            hd95[i] = 0.0
        elif gt_empty and not pred_empty:
            pred_array = pred[i].cpu().numpy()
            if np.sum(pred_array) > 0:
                com = center_of_mass(pred_array)
                com_mask = np.zeros_like(pred_array, dtype=np.uint8)
                com_coords = tuple(map(int, map(round, com)))
                com_mask[com_coords] = 1
                com_mask_tensor = torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                mock_val = compute_hausdorff_distance(
                    y_pred=torch.stack(pred)[i].unsqueeze(0),
                    y=com_mask_tensor.unsqueeze(0),
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95,
                )
                print(f"Mock HD95 for region {i} (GT absent):", mock_val.item())
                print(f"Before update, hd95: {hd95}")
                hd95[i] = mock_val.item()
                print(f"After update, hd95: {hd95}")
            else:
                hd95[i] = 0.0
        elif pred_empty and not gt_empty:
            gt_array = torch.stack(gt)[i].cpu().numpy()
            if np.sum(gt_array) > 0:
                com = center_of_mass(gt_array)
                com_mask = np.zeros_like(gt_array, dtype=np.uint8)
                com_coords = tuple(map(int, map(round, com)))
                com_mask[com_coords] = 1
                com_mask_tensor = torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                mock_val = compute_hausdorff_distance(
                    y_pred=torch.stack(gt)[i].unsqueeze(0),
                    y=com_mask_tensor.unsqueeze(0),
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95,
                )
                print(f"Mock HD95 for region {i} (Prediction absent):", mock_val.item())
                print(f"Before update, hd95: {hd95}")
                hd95[i] = mock_val.item()
                print(f"After update, hd95: {hd95}")
            else:
                print(f"Warning: GT mask for region {i} is unexpectedly empty.")
                hd95[i] = 0.0
    confusion_metric(y_pred=pred, y=gt)
    sensitivity, specificity = confusion_metric.aggregate()
    sensitivity = sensitivity.squeeze(0).cpu().numpy()
    specificity = specificity.squeeze(0).cpu().numpy()
    for i in range(len(sensitivity)):
        if not_nans[i] == 0:
            pred_empty = torch.sum(pred_stack[i]).item() == 0
            sensitivity[i] = 1.0 if pred_empty else 0.0
            specificity[i] = 1.0
    return dice_scores, hd95, sensitivity, specificity

########################################
#### Perform Ensemble Segmentation ####
########################################

def extract_patient_id(path):
    numbers = re.findall("\d+", path)
    patient_id = numbers[-1]
    return patient_id

def save_metrics_csv(metrics_list, filename):
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)
    print(f"Saved patient-wise metrics to {filename}")

def save_average_metrics(metrics_list, filename):
    avg_metrics = {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
        if key != "patient_id"
    }
    with open(filename, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Saved average test set metrics to {filename}")

def save_uncertainty_as_nifti(uncertainty_map, ref_img, output_path):
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.cpu().numpy()
    uncertainty_map = minmax_uncertainties(uncertainty_map)
    uncertainty_map = uncertainty_map.astype(np.float32)
    uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=ref_img.affine, header=ref_img.header)
    nib.save(uncertainty_nifti, output_path)
    print(f"Uncertainty map saved to {output_path}")

def compute_disagreement_uncertainty(probability_maps, weights=None):
    probability_maps = np.array(probability_maps)
    if weights is not None:
        weights = np.array(weights).reshape(-1, 1, 1, 1, 1)
        mean_prob = np.sum(weights * probability_maps, axis=0)
        variance = np.sum(weights * (probability_maps - mean_prob)**2, axis=0)
    else:
        mean_prob = np.mean(probability_maps, axis=0)
        variance = np.var(probability_maps, axis=0)
    return variance

def compute_hard_label_disagreement(hard_label_maps):
    """
    Compute voxel-wise disagreement for a set of binary hard label maps
    using Shannon entropy.
    
    Args:
        hard_label_maps (list of np.array): A list of binary (0/1) maps of shape (H, W, D)
            where 1 indicates the presence of the region and 0 its absence.
    
    Returns:
        np.array: A disagreement map of shape (H, W, D) where higher values indicate higher disagreement.
    """
    # Convert list to numpy array with shape [N_models, H, W, D]
    preds = np.array(hard_label_maps)
    # Compute the fraction of models that predict the region (i.e. vote = 1)
    p = np.mean(preds, axis=0)  # p has shape (H, W, D) and values between 0 and 1
    eps = 1e-6  # small constant to avoid log(0)
    # Compute entropy (uncertainty) at each voxel:
    # Note: The entropy is maximum when p is 0.5 and zero when p is 0 or 1.
    disagreement = - (p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
    return disagreement

def ensemble_segmentation(
    test_loader,
    models_dict,
    composite_score_weights,
    n_iterations=10,
    patient_id=None,
    output_dir="./output_segmentations/ttd",
):
    """
    Perform segmentation using an ensemble of multiple models.
    """
    os.makedirs(output_dir, exist_ok=True)
    if patient_id is not None:
        test_data_loader = config.find_patient_by_id(patient_id, test_loader)
    else:
        test_data_loader = test_loader

    # Compute performance weights for all classes (BG, NCR, ED, ET)
    model_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
    for model_name in models_dict.keys():
        performance_weights_path = f"../models/performance/{model_name}/average_metrics.json"
        metrics = load_weights(performance_weights_path)
        composite_scores = compute_composite_scores(metrics, composite_score_weights)
        for region in ["BG", "NCR", "ED", "ET"]:
            model_weights[region][model_name] = composite_scores[region]
    # Normalize (the normalization here remains scalar per region)
    for region in ["BG", "NCR", "ED", "ET"]:
        total_weight = sum(model_weights[region].values())
        for model_name in model_weights[region]:
            model_weights[region][model_name] /= total_weight
    print(f"Computed model performance weights per class: {model_weights}")

    patient_metrics = []
    with torch.no_grad():
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            ref_img = nib.load(reference_image_path)
            patient_id = extract_patient_id(reference_image_path)
            gt = batch_data["label"].to(config.device)
            print(f"\nProcessing patient: {patient_id}\n")
            
            # Prepare dictionaries to store per-model adjusted (voxel-level weighted) predictions
            # and model uncertainties.
            model_predictions = {}
            model_uncertainties = {}   # NEW: dictionary to save each model's uncertainty

            # STEP 1: For each model, get uncertainties and predictions,
            # then compute a voxel-level weighted prediction.
            for model_name, (model, inferer) in models_dict.items():
                ttd_mean, ttd_uncertainty = ttd_variance(
                    inferer, image, config.device, n_iterations=n_iterations
                )

                eps = 1e-6  # small constant to avoid division by zero
                # Compute voxel-level inverse uncertainty maps (used for weighting the prediction)
                inv_ttd = 1.0 / (ttd_uncertainty + eps)

                # then further weight by the inverse TTD uncertainty.
                adjusted_prediction = ttd_mean * inv_ttd
                # Remove extra dimensions if necessary:
                adjusted_prediction = np.squeeze(adjusted_prediction)  # expected shape: [num_classes, H, W, D]

                model_predictions[model_name] = adjusted_prediction

                # Compute a combined uncertainty for the current model.
                model_uncertainties[model_name] = np.squeeze(ttd_uncertainty)

            # STEP 2: Fuse predictions from all models.
            # Here we still use the scalar performance (composite) weights computed earlier.
            weighted_logits = {region: [] for region in ["BG", "NCR", "ED", "ET"]}
            for model_name in models_dict.keys():
                # Convert each model’s adjusted prediction (logits) to a torch tensor.
                logits = torch.from_numpy(model_predictions[model_name]).to(config.device)
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    # Get the performance weight scalar (previously computed and normalized)
                    perf_weight = model_weights[region][model_name]
                    # Multiply each voxel’s logit by the performance weight:
                    region_logits = logits[idx]
                    weighted_region_logits = perf_weight * region_logits
                    weighted_logits[region].append(weighted_region_logits)
            # Sum the weighted logits across models.
            fused_bg = torch.sum(torch.stack(weighted_logits["BG"]), dim=0)
            fused_tumor = [torch.sum(torch.stack(weighted_logits[region]), dim=0) for region in ["NCR", "ED", "ET"]]
            fused_logits = torch.stack([fused_bg] + fused_tumor, dim=0)

            # Convert logits to probability maps and generate segmentation.
            fused_probs = torch.softmax(fused_logits, dim=0)
            seg = fused_probs.argmax(dim=0).unsqueeze(0)

            # Prepare one-hot representations.
            pred_one_hot = [(seg == i).float() for i in range(0, 4)]
            if gt.shape[1] == 4:
                gt_one_hot = gt.permute(1, 0, 2, 3, 4)
            else:
                gt_one_hot = [(gt == i).float() for i in range(0, 4)]
                gt_one_hot = torch.stack(gt_one_hot)

            # Compute performance metrics.
            dice, hd95, sensitivity, specificity = compute_metrics(pred_one_hot, gt_one_hot)
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

            adjusted_weights = {region: model_weights[region] for region in ["NCR", "ED", "ET"]}

            # Fuse uncertainty maps for tumor regions using the saved model_uncertainties.
            fused_uncertainty = {}
            for idx, region in enumerate(["NCR", "ED", "ET"]):
                # Use one model's uncertainty shape to create an initial zero tensor.
                uncertainty_sum = torch.zeros_like(torch.from_numpy(model_uncertainties[next(iter(model_uncertainties))][idx+1]))
                for model_name in model_uncertainties:
                    weight = adjusted_weights[region][model_name]
                    # Here we assume that model_uncertainties[model_name] has shape [num_classes, H, W, D]
                    # and region indices 1,2,3 correspond to "NCR", "ED", "ET"
                    uncertainty_sum += weight * torch.from_numpy(model_uncertainties[model_name][idx+1])
                fused_uncertainty[region] = minmax_uncertainties(uncertainty_sum.cpu().numpy())

            disagreement_uncertainty = {}
            for idx, region in enumerate(["NCR", "ED", "ET"]):
                region_prob_maps = []
                for model_name in models_dict.keys():
                    logits = torch.from_numpy(model_predictions[model_name]).to(config.device)
                    region_logits = logits[idx+1]
                    prob_map = torch.softmax(region_logits, dim=0).cpu().numpy()
                    region_prob_maps.append(prob_map)
                disagreement_uncertainty[region] = compute_disagreement_uncertainty(region_prob_maps)
                
            disagreement_uncertainty_hl = {}
            for idx, region in enumerate(["NCR", "ED", "ET"]):
                hard_label_maps = []
                for model_name in models_dict.keys():
                    # Convert model predictions (assumed to be a numpy array) to a tensor
                    logits = torch.from_numpy(model_predictions[model_name]).to(config.device)
                    # Compute the hard segmentation mask for the entire image (argmax over channels)
                    # This results in a tensor of shape [H, W, D] with labels 0 (BG), 1 (NCR), 2 (ED), 3 (ET)
                    seg = torch.argmax(logits, dim=0)
                    # For the current region, create a binary mask (1 where the label equals idx+1)
                    hard_map = (seg == (idx + 1)).cpu().numpy().astype(np.uint8)
                    hard_label_maps.append(hard_map)
                # Compute the disagreement map using the helper function defined above
                disagreement_uncertainty_hl[region] = compute_hard_label_disagreement(hard_label_maps)
                
            for region in ["NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}_fused.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}_disagreement.nii.gz")
                save_uncertainty_as_nifti(disagreement_uncertainty[region], ref_img, output_path)
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}_disagreement_hl.nii.gz")
                save_uncertainty_as_nifti(disagreement_uncertainty_hl[region], ref_img, output_path)

            output_path = os.path.join(output_dir, f"ttd_softmax_{patient_id}.nii.gz")
            save_probability_map_as_nifti(fused_probs, ref_img, output_path)

            output_path = os.path.join(output_dir, f"ttd_segmentation_{patient_id}.nii.gz")
            seg = seg.squeeze(0)
            save_segmentation_as_nifti(seg, reference_image_path, output_path)

            torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, "ttd_patient_metrics_test.csv")
    json_path = os.path.join(output_dir, "ttd_average_metrics_test.json")
    save_metrics_csv(patient_metrics, csv_path)
    save_average_metrics(patient_metrics, json_path)

#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "01556"
    models_dict = load_all_models()
    composite_score_weights = {
        "Dice": 0.45,
        "HD95": 0.15,
        "Sensitivity": 0.3,
        "Specificity": 0.1,
    }
    ensemble_segmentation(
        config.test_loader, models_dict, composite_score_weights, n_iterations=10
    )
