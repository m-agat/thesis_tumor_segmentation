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
import pandas as pd
import json
import re
from torch.utils.data import Subset
from sklearn.linear_model import LogisticRegression
from calibration_evaluation import compute_ece, compute_reliability_data, plot_reliability_diagram
import pickle

# Add custom modules
sys.path.append("../")
import models.models as models
import config.config as config
from uncertainty.test_time_augmentation import tta_variance
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
        "vnet": load_model(
            models.vnet_model, config.model_paths["vnet"], config.device
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

def save_segmentation_as_nifti(
    predicted_segmentation, reference_image_path, output_path
):
    """
    Save the predicted segmentation as a NIfTI file.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()

    predicted_segmentation = predicted_segmentation.astype(np.uint8)
    ref_img = nib.load(reference_image_path)
    seg_img = nib.Nifti1Image(
        predicted_segmentation, affine=ref_img.affine, header=ref_img.header
    )
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
    dice_metric = DiceMetric(
        include_background=False, reduction=MetricReduction.NONE, get_not_nans=True
    )
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity"],
        reduction="none",
        compute_sample=False,
    )

    # Convert MetaTensors to plain tensors if needed.
    pred = [p.detach().clone() if hasattr(p, "detach") else p for p in pred]
    gt = [g.detach().clone() if hasattr(g, "detach") else g for g in gt]

    pred_stack = torch.stack(pred)
    gt_stack = torch.stack(gt)

    # Compute Dice Scores
    dice_metric(y_pred=pred, y=gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.cpu().numpy()

    for i, dice_score in enumerate(dice_scores):
        if not_nans[i] == 0:  # Tissue is absent in ground truth
            pred_empty = torch.sum(pred_stack[i]).item() == 0
            dice_scores[i] = 1.0 if pred_empty else 0.0

    # Compute HD95
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

        elif gt_empty and not pred_empty:  # Ground truth is absent.
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

        elif pred_empty and not gt_empty:  # Model predicts tissue is absent
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

    # Compute Sensitivity & Specificity
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
    # Use regular expression to find all numbers in the path
    numbers = re.findall("\d+", path)
    # Assuming the patient ID is the last number found
    patient_id = numbers[-1]
    return patient_id

def save_metrics_csv(metrics_list, filename):
    """
    Save per-patient segmentation performance metrics to CSV.
    """
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)
    print(f"Saved patient-wise metrics to {filename}")

def save_average_metrics(metrics_list, filename):
    """
    Save the average test set performance in a JSON file.
    """
    avg_metrics = {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
        if key != "patient_id"
    }
    with open(filename, "w") as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Saved average test set metrics to {filename}")

def save_uncertainty_as_nifti(uncertainty_map, ref_img, output_path):
    """
    Save a 3D uncertainty map as a NIfTI file with optional scaling.
    """
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.cpu().numpy()
    uncertainty_map = minmax_uncertainties(uncertainty_map)
    uncertainty_map = uncertainty_map.astype(np.float32)
    uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=ref_img.affine, header=ref_img.header)
    nib.save(uncertainty_nifti, output_path)
    print(f"Uncertainty map saved to {output_path}")

def evaluate_calibration(uncs, errs, region, n_bins=10, save_prefix="raw"):
    """
    Compute and plot reliability diagram and ECE.
    """
    bin_centers, mean_conf, acc = compute_reliability_data(uncs, errs, n_bins=n_bins)
    bins = np.linspace(0, 1, n_bins + 1)
    error_counts = np.array([np.sum((uncs >= bins[i]) & (uncs < bins[i+1])) for i in range(n_bins)])
    n_total = uncs.size
    ece = compute_ece(mean_conf, acc, error_counts, n_total)
    
    plot_reliability_diagram(
        bin_centers,
        mean_conf,
        acc,
        title=f"Reliability Diagram ({region}, {save_prefix})",
        save_path=f"reliability_diagram_{region}_{save_prefix}.png"
    )
    print(f"ECE for {region} ({save_prefix}): {ece:.3f}")
    return ece

def fit_logistic_calibrators_with_evaluation(val_loader, models_dict, n_iterations=10, device="cuda"):
    """
    1) Runs inference on the val_loader using your ensemble TTA+TTD logic.
    2) Collects (uncertainty, error) pairs for each region.
    3) Evaluates calibration before and after fitting logistic regression.
    4) Trains a logistic regression calibrator for each region.
    
    Returns:
        calibrators : dict of trained LogisticRegression models.
    """
    # Define region names (BG, NCR, ED, ET)
    region_names = ["BG", "NCR", "ED", "ET"]
    all_uncs = {r: [] for r in region_names}
    all_errs = {r: [] for r in region_names}

    print("Performing uncertainty calibration...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            image = batch_data["image"].to(device)
            gt = batch_data["label"].to(device)
            reference_image_path = batch_data["path"][0]
            patient_id = extract_patient_id(reference_image_path)

            print(f"Processing patient {batch_idx+1}/{len(val_loader)} - {patient_id}")
            
            ensemble_preds = []
            ensemble_uncs = []
            
            for model_name, (model, inferer) in models_dict.items():
                ttd_mean, ttd_uncertainty = ttd_variance(model, inferer, image, device, n_iterations=n_iterations)
                tta_mean, tta_uncertainty = tta_variance(inferer, image, device, n_iterations=n_iterations)
                hybrid_mean = 0.5 * (tta_mean + ttd_mean)
                hybrid_uncertainty = 0.5 * (tta_uncertainty + ttd_uncertainty)
                hybrid_uncertainty = minmax_uncertainties(hybrid_uncertainty)
                ensemble_preds.append(hybrid_mean)
                ensemble_uncs.append(hybrid_uncertainty)
            
            fused_pred = np.mean(np.stack(ensemble_preds, axis=0), axis=0)
            fused_unc  = np.mean(np.stack(ensemble_uncs, axis=0), axis=0)
            
            if not isinstance(fused_pred, torch.Tensor):
                fused_pred_tensor = torch.from_numpy(fused_pred)
            else:
                fused_pred_tensor = fused_pred
            seg = fused_pred_tensor.argmax(dim=1, keepdim=True)
            
            seg_np = seg.cpu().numpy().squeeze(0).squeeze(0)
            gt_seg = gt.argmax(dim=1)
            gt_np = gt_seg.cpu().numpy().squeeze(0)
            # Assume fused_unc has shape [4, H, W, D] corresponding to BG, NCR, ED, ET.
            unc_np = np.squeeze(fused_unc, axis=0)
            
            # Instead of using the ground truth mask, we use the model's prediction mask.
            for idx, region in enumerate(region_names):
                # Use all voxels in the predicted mask
                predicted_mask = (seg_np == idx)
                region_unc = unc_np[idx][predicted_mask]
                # Define error = 1 if ground truth disagrees with the predicted label
                region_err = (gt_np[predicted_mask] != idx).astype(np.uint8)
                
                all_uncs[region].append(region_unc)
                all_errs[region].append(region_err)
    
    calibrators = {}
    for region in region_names:
        uncs_cat = np.concatenate(all_uncs[region], axis=0)
        errs_cat = np.concatenate(all_errs[region], axis=0)
        uncs_cat = uncs_cat.reshape(-1, 1)
        errs_cat = errs_cat.ravel()
        
        print(f"Before calibration for {region}:")
        evaluate_calibration(uncs_cat.flatten(), errs_cat, region, save_prefix="raw")
        
        # Use all voxels and add class_weight="balanced"
        lr = LogisticRegression(class_weight="balanced")
        lr.fit(uncs_cat, errs_cat)
        calibrators[region] = lr
        
        # Apply calibration to the same uncertainties.
        calibrated_uncs = lr.predict_proba(uncs_cat)[:, 1]
        print(f"After calibration for {region}:")
        evaluate_calibration(calibrated_uncs.flatten(), errs_cat, region, save_prefix="calibrated")
    
    with open("calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)  # 1 calibrator per region (total 4 calibrators)

    return calibrators

def ensemble_segmentation(
    test_loader,
    models_dict,
    composite_score_weights,
    n_iterations=10,
    calibrators=None,  
    patient_id=None,
    output_dir="./output_segmentations/calibrated",
):
    """
    Perform segmentation using an ensemble of multiple models with simple averaging.
    Calibration is applied to the raw uncertainties before computing ensemble weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    if patient_id is not None:
        # Get a subset of test loader with the specific patient
        test_data_loader = config.find_patient_by_id(patient_id, test_loader)
    else:
        # Get full test data loader
        test_data_loader = test_loader

    # Compute performance weights for all classes (BG, NCR, ED, ET)
    model_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
    for model_name in models_dict.keys():
        performance_weights_path = (
            f"../models/performance/{model_name}/average_metrics.json"
        )
        metrics = load_weights(performance_weights_path)
        composite_scores = compute_composite_scores(metrics, composite_score_weights)
        for region in ["BG", "NCR", "ED", "ET"]:
            model_weights[region][model_name] = composite_scores[region]
    # Normalize weights per class
    for region in ["BG", "NCR", "ED", "ET"]:
        total_weight = sum(model_weights[region].values())
        for model_name in model_weights[region]:
            model_weights[region][model_name] /= total_weight

    print(f"Computed model performance weights per class: {model_weights}")

    patient_metrics = []
    with torch.no_grad():
        for idx, batch_data in enumerate(test_data_loader):
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            ref_img = nib.load(reference_image_path)
            patient_id = extract_patient_id(reference_image_path)
            gt = batch_data["label"].to(config.device)  # shape: (B, H, W, D)

            print(f"\nProcessing patient {idx+1}/{len(test_data_loader)}: {patient_id}\n")

            w_ttd = 0.5
            w_tta = 0.5
            adjusted_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
            model_predictions = {}
            model_uncertainties = {}

            # Step 1: Compute adjusted weights for each model.
            for model_name, (model, inferer) in models_dict.items():
                ttd_mean, ttd_uncertainty = ttd_variance(
                    model, inferer, image, config.device, n_iterations=n_iterations
                )
                tta_mean, tta_uncertainty = tta_variance(
                    inferer, image, config.device, n_iterations=n_iterations
                )

                hybrid_mean = (ttd_mean + tta_mean) / 2
                hybrid_mean = np.squeeze(hybrid_mean)
                raw_hybrid_uncertainty = w_ttd * ttd_uncertainty + w_tta * tta_uncertainty
                raw_hybrid_uncertainty = np.squeeze(raw_hybrid_uncertainty, axis=0)

                # Apply calibration to each region's uncertainty (if a calibrator exists)
                calibrated_uncertainty = raw_hybrid_uncertainty.copy()
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    if calibrators is not None and region in calibrators:
                        flat_unc = raw_hybrid_uncertainty[idx].ravel().reshape(-1, 1)
                        calibrated_flat = calibrators[region].predict_proba(flat_unc)[:, 1]
                        calibrated_uncertainty[idx] = calibrated_flat.reshape(raw_hybrid_uncertainty[idx].shape)
                model_predictions[model_name] = hybrid_mean
                model_uncertainties[model_name] = calibrated_uncertainty

                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    alpha = 1
                    uncertainty_penalty = (1 - alpha * np.median(calibrated_uncertainty[idx]))
                    print(f"Model {model_name}, Region {region}: calibrated uncertainty median = {np.median(calibrated_uncertainty[idx])}, penalty = {uncertainty_penalty}")
                    adjusted_weights[region][model_name] = model_weights[region][model_name] * uncertainty_penalty
                    print("Adjusted weight: ", adjusted_weights[region][model_name])

            for region in ["BG", "NCR", "ED", "ET"]:
                total_weight = sum(adjusted_weights[region].values())
                for model_name in adjusted_weights[region]:
                    adjusted_weights[region][model_name] /= total_weight
                    print(f"Model: {model_name}, Region: {region}, Final Weight: {adjusted_weights[region][model_name]:.3f}")

            weighted_probs = {region: [] for region in ["BG", "NCR", "ED", "ET"]}
            for model_name in models_dict.keys():
                probs = torch.from_numpy(model_predictions[model_name]).to(config.device)
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    weight = torch.tensor(adjusted_weights[region][model_name], dtype=torch.float32, device=config.device)
                    region_prob = probs[idx]
                    weighted_probs[region].append(weight * region_prob)

            fused_background = torch.sum(torch.stack(weighted_probs["BG"]), dim=0)
            fused_tumor = [torch.sum(torch.stack(weighted_probs[region]), dim=0) for region in ["NCR", "ED", "ET"]]
            fused_probs = torch.stack([fused_background] + fused_tumor, dim=0)
            seg = fused_probs.argmax(dim=0).unsqueeze(0)

            pred_one_hot = [(seg == i).float() for i in range(0, 4)]
            if gt.shape[1] == 4:
                gt_one_hot = gt.permute(1, 0, 2, 3, 4)
            else:
                gt_one_hot = [(gt == i).float() for i in range(0, 4)]
                gt_one_hot = torch.stack(gt_one_hot)

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

            fused_uncertainty = {}
            uncertainty_sum_bg = torch.zeros_like(torch.from_numpy(model_uncertainties[next(iter(model_uncertainties))][0]))
            for model_name in model_uncertainties:
                weight_bg = adjusted_weights["BG"][model_name]
                uncertainty_sum_bg += weight_bg * torch.from_numpy(model_uncertainties[model_name][0])
            fused_uncertainty["BG"] = uncertainty_sum_bg.cpu().numpy()

            for idx, region in enumerate(["NCR", "ED", "ET"]):
                uncertainty_sum = torch.zeros_like(torch.from_numpy(model_uncertainties[next(iter(model_uncertainties))][idx+1]))
                for model_name in model_uncertainties:
                    weight = adjusted_weights[region][model_name]
                    uncertainty_sum += weight * torch.from_numpy(model_uncertainties[model_name][idx+1])
                fused_uncertainty[region] = uncertainty_sum.cpu().numpy()
            
            for region in ["BG", "NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}_calibrated.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)

            pred_seg = seg.squeeze(0).cpu().numpy()
            region_index_dict = {"BG": 0, "NCR": 1, "ED": 2, "ET": 3}
            for region in ["BG", "NCR", "ED", "ET"]:
                mask = (pred_seg == region_index_dict[region])
                fused_uncertainty[region] = fused_uncertainty[region] * mask.astype(np.float32)

            for region in ["BG", "NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}_viz.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)

            output_path = os.path.join(
                output_dir, f"calibrated_softmax_{patient_id}_new.nii.gz"
            )
            softmax_seg = torch.nn.functional.softmax(fused_probs, dim=0).squeeze(0)
            save_probability_map_as_nifti(softmax_seg, ref_img, output_path)

            output_path = os.path.join(
                output_dir, f"calibrated_segmentation_{patient_id}_new.nii.gz"
            )
            seg = seg.squeeze(0)
            save_segmentation_as_nifti(seg, reference_image_path, output_path)

            torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, "calibrated_patient_metrics.csv")
    json_path = os.path.join(output_dir, "calibrated_average_metrics.json")
    save_metrics_csv(patient_metrics, csv_path)
    save_average_metrics(patient_metrics, json_path)

#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "00483"
    models_dict = load_all_models()
    composite_score_weights = {
        "Dice": 0.45,
        "HD95": 0.15,
        "Sensitivity": 0.3,
        "Specificity": 0.1,
    }

    _, val_loader = dataloaders.get_loaders(
        batch_size=config.batch_size,
        json_path=config.json_path,
        basedir=config.root_dir,
        fold=None,
        roi=config.roi,
        use_final_split=True,
    )
    full_dataset = val_loader.dataset
    all_indices = list(range(len(full_dataset)))
    print("all ", len(all_indices))

    val_loader_subset = config.create_subset(val_loader, 5, shuffle=False)
    calib_indices = val_loader_subset.dataset.indices 
    print("calib ", len(calib_indices))

    eval_indices = [idx for idx in all_indices if idx not in calib_indices]
    print("eval ", len(eval_indices))

    val_subset_eval = Subset(full_dataset, eval_indices)
    val_loader_subset_eval = torch.utils.data.DataLoader(
        val_subset_eval,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=True,
    )
    
    calibrators = fit_logistic_calibrators_with_evaluation(
        val_loader_subset,
        models_dict,
        n_iterations=5,
        device=config.device,
    )

    ensemble_segmentation(
        val_loader_subset_eval, models_dict, composite_score_weights, n_iterations=5, calibrators=calibrators, patient_id=patient_id
    )
