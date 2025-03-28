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
from uncertainty.test_time_augmentation import tta_variance, tta_entropy
from uncertainty.test_time_dropout import ttd_variance, minmax_uncertainties, ttd_entropy
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

    Parameters:
    - predicted_segmentation: The segmentation output as a tensor or numpy array.
    - reference_image_path: Path to the reference NIfTI image (for affine and header copying).
    - output_path: Path where the new segmentation NIfTI file will be saved.
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
    # Make sure to keep the float values
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
        # Use the i-th class mask directly.
        pred_empty = torch.sum(pred[i]).item() == 0
        gt_empty = not_nans[i] == 0

        if pred_empty and gt_empty:
            print(f"Region {i}: Both GT and Prediction are empty. Setting HD95 to 0.")
            hd95[i] = 0.0

        elif gt_empty and not pred_empty:  # Ground truth is absent.
            pred_array = pred[i].cpu().numpy()  # Use pred[i] directly.
            if np.sum(pred_array) > 0:
                # Compute Center of Mass for the predicted mask
                com = center_of_mass(pred_array)
                com_mask = np.zeros_like(pred_array, dtype=np.uint8)
                com_coords = tuple(
                    map(int, map(round, com))
                )  # Round and convert to integer indices
                com_mask[com_coords] = 1

                # Convert CoM mask back to tensor
                com_mask_tensor = (
                    torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                )

                # Compute Hausdorff Distance between prediction and CoM mask
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
                # No prediction or GT; HD95 = 0
                hd95[i] = 0.0

        elif pred_empty and not gt_empty:  # Model predicts tissue is absent
            gt_array = torch.stack(gt)[i].cpu().numpy()
            if np.sum(gt_array) > 0:
                # Compute Center of Mass for the GT mask
                com = center_of_mass(gt_array)
                com_mask = np.zeros_like(gt_array, dtype=np.uint8)
                com_coords = tuple(
                    map(int, map(round, com))
                )  # Round and convert to integer indices
                com_mask[com_coords] = 1

                # Convert CoM mask back to tensor
                com_mask_tensor = (
                    torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                )

                # Compute Hausdorff Distance between GT CoM and empty prediction
                mock_val = compute_hausdorff_distance(
                    y_pred=torch.stack(gt)[i].unsqueeze(0),
                    y=com_mask_tensor.unsqueeze(0),
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95,
                )

                print(
                    f"Mock HD95 for region {i} (Prediction absent):",
                    mock_val.item(),
                )
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
        if not_nans[i] == 0:  # Tissue is absent
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


def ensemble_segmentation(
    test_loader,
    models_dict,
    composite_score_weights,
    n_iterations=10,
    patient_id=None,
    output_dir="./output_segmentations/hybrid",
):
    """
    Perform segmentation using an ensemble of multiple models with simple averaging.

    Parameters:
    - patient_id: ID of the patient whose scan is being segmented.
    - test_loader: Dataloader for the test set.
    - models_dict: Dictionary containing trained models and their inferers.
    - output_dir: Directory where segmentations will be saved.
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
            if model_name in ["attunet", "vnet"]:
                model_weights[region][model_name] /= total_weight
            else:
                # reduce influence of segresnet and swinunetr ensuring that attunet and vnet will also meaningfully contribute
                model_weights[region][model_name] /= total_weight

    print(f"Computed model performance weights per class: {model_weights}")

    patient_metrics = []
    with torch.no_grad():
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            ref_img = nib.load(reference_image_path)
            patient_id = extract_patient_id(reference_image_path)
            gt = batch_data["label"].to(
                config.device
            )  # shape: (batch_size, 240, 240, 155)

            print(
                f"\nProcessing patient: {patient_id}\n",
            )

            w_ttd = 0.5
            w_tta = 0.5
            adjusted_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
            model_predictions = {}
            model_uncertainties = {}            
            # Step 1: Compute adjusted weights for AttUNet & VNet (Performance × Uncertainty Scaling)
            # Inside the ensemble_segmentation loop, for each model:
            for model_name, (model, inferer) in models_dict.items():
                # Get dropout uncertainty (TTD) and augmentation uncertainty (TTA) with entropy
                ttd_mean, ttd_uncertainty = ttd_entropy(
                    model, image, inferer, config.device, n_iterations=n_iterations
                )
                tta_mean, tta_uncertainty = tta_entropy(
                    inferer, image, config.device, n_iterations=n_iterations
                )

                # Compute the hybrid mean (probability map) and hybrid uncertainty (global entropy map)
                hybrid_mean = (ttd_mean + tta_mean) / 2  # Expected shape: (C, H, W, D)
                hybrid_mean = np.squeeze(hybrid_mean)    # Now shape: (C, H, W, D) after removing batch dim
                hybrid_uncertainty = w_ttd * ttd_uncertainty + w_tta * tta_uncertainty  # Shape: (H, W, D)

                # Compute a tissue-specific prediction from the probability map
                # Assuming channels: 0=BG, 1=NCR, 2=ED, 3=ET.
                pred_seg = np.argmax(hybrid_mean, axis=0)  # Shape: (H, W, D)

                model_predictions[model_name] = hybrid_mean
                model_uncertainties[model_name] = hybrid_uncertainty

                # Instead of indexing hybrid_uncertainty by channel, mask it by the tissue labels
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    # Create a mask for the current tissue based on the predicted segmentation.
                    region_mask = (pred_seg == idx)
                    # Compute the median entropy only over voxels predicted as the current tissue.
                    if np.sum(region_mask) > 0:
                        median_uncertainty = np.nanmedian(hybrid_uncertainty[region_mask])
                    else:
                        median_uncertainty = 0.0

                    alpha = 0.5
                    uncertainty_penalty = (1 - alpha * median_uncertainty)
                    print(f"Model: {model_name}, Region: {region}, Median Entropy: {median_uncertainty:.4f}, Penalty: {uncertainty_penalty:.4f}")

                    adjusted_weights[region][model_name] = model_weights[region][model_name] * uncertainty_penalty


            # Step 2: Normalize weights per region
            for region in ["BG", "NCR", "ED", "ET"]:
                total_weight = sum(adjusted_weights[region].values())
                for model_name in adjusted_weights[region]:
                    adjusted_weights[region][model_name] /= total_weight
                    print(
                            f"Model: {model_name}, Region: {region}, Final Weight: {adjusted_weights[region][model_name]:.3f}"
                        )

            # Compute weighted logits fusion as before and also store raw logits for uncertainty computation.
            weighted_logits = {region: [] for region in ["BG", "NCR", "ED", "ET"]}
            ensemble_logits_list = []  # to store each model's raw logits for uncertainty computation

            for model_name in models_dict.keys():
                # Convert the stored NumPy prediction to a torch tensor and move to the device.
                logits = torch.from_numpy(model_predictions[model_name]).to(config.device)  # shape: [num_classes, H, W, D]
                ensemble_logits_list.append(logits)
                
                # Apply normalized final weights for each region.
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    weight = torch.tensor(adjusted_weights[region][model_name],
                                          dtype=torch.float32, device=config.device)
                    logits_tensor = logits[idx]
                    weighted_logits[region].append(weight * logits_tensor)

            # Fuse logits: weighted sum for each class.
            fused_background = torch.sum(torch.stack(weighted_logits["BG"]), dim=0)
            fused_tumor = [
                torch.sum(torch.stack(weighted_logits[region]), dim=0)
                for region in ["NCR", "ED", "ET"]
            ]
            fused_logits = torch.stack([fused_background] + fused_tumor, dim=0)
            softmax_seg = torch.nn.functional.softmax(fused_logits, dim=0)
            seg = softmax_seg.argmax(dim=0).unsqueeze(0)

            pred_one_hot = [(seg == i).float() for i in range(0, 4)]
            if gt.shape[1] == 4:
                # Ground truth is already one-hot encoded (assume channel 0 is background).
                # Extract channels 1,2,3 and permute to have shape [3, 1, H, W, D].
                gt_one_hot = gt.permute(1, 0, 2, 3, 4)
            else:
                # Ground truth is not one-hot encoded, so create one-hot encoding.
                gt_one_hot = [(gt == i).float() for i in range(0, 4)]
                gt_one_hot = torch.stack(gt_one_hot)

            # Get performance metrics
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

            # Fuse uncertainty maps for tumor regions
            seg = seg.squeeze(0).cpu().numpy()
            fused_uncertainty = {}
            for region, label in zip(["NCR", "ED", "ET"], [1, 2, 3]):
                # Fuse the global uncertainty maps using the adjusted weights (note: here each model provides a global map).
                uncertainty_sum = np.zeros_like(model_uncertainties[next(iter(model_uncertainties))])
                for model_name in model_uncertainties:
                    uncertainty_sum += adjusted_weights[region][model_name] * model_uncertainties[model_name]
                # Mask the fused uncertainty to only include voxels labeled as the current tissue in the ensemble segmentation.
                tissue_mask = (seg == label)
                regional_uncertainty = np.where(tissue_mask, uncertainty_sum, 0)
                fused_uncertainty[region] = minmax_uncertainties(regional_uncertainty)

            for region in ["NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"entropy_uncertainty_{region}_{patient_id}.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)

            output_path = os.path.join(
                output_dir, f"hybrid_softmax_{patient_id}_entropy.nii.gz"
            )
            softmax_seg = softmax_seg.squeeze(0)  # remove batch dimension
            save_probability_map_as_nifti(softmax_seg, ref_img, output_path)

            output_path = os.path.join(
                output_dir, f"hybrid_segmentation_{patient_id}_entropy.nii.gz"
            )
              # remove batch dimension
            save_segmentation_as_nifti(seg, reference_image_path, output_path)

            torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, "hybrid_patient_metrics_entropy.csv")
    json_path = os.path.join(output_dir, "hybrid_average_metrics_entropy.json")
    save_metrics_csv(patient_metrics, csv_path)
    save_average_metrics(patient_metrics, json_path)


####################################
#### Visualize Segmentation ####
####################################


def visualize_segmentation(segmentation, patient_id):
    """
    Display and save a middle slice from the segmentation.

    Parameters:
    - segmentation: 3D NumPy array containing the segmentation result.
    - patient_id: ID of the patient (used for file naming).
    """
    slice_index = segmentation.shape[-1] // 2  # Middle slice

    plt.figure(figsize=(6, 6))
    plt.imshow(segmentation[:, :, slice_index], cmap="gray")
    plt.title(f"Segmentation Slice at Index {slice_index}")
    plt.axis("off")
    plt.savefig(f"hybrid_segmentation_{patient_id}_slice.png")
    # plt.show()


#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "00113"
    models_dict = load_all_models()
    _, val_loader = dataloaders.get_loaders(
        batch_size=config.batch_size,
        json_path=config.json_path,
        basedir=config.root_dir,
        fold=None,
        roi=config.roi,
        use_final_split=True,
    )
    composite_score_weights = {
        "Dice": 0.45,
        "HD95": 0.15,
        "Sensitivity": 0.3,
        "Specificity": 0.1,
    }
    ensemble_segmentation(
        val_loader, models_dict, composite_score_weights, n_iterations=10, patient_id=patient_id
    )
