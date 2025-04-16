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
import models.models as models
import config.config as config
from utils.utils import AverageMeter

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
        pred_empty = torch.sum(torch.stack(pred), dim=[1, 2, 3, 4])[i].item() == 0
        gt_empty = not_nans[i] == 0

        if pred_empty and gt_empty:
            print(f"Region {i}: Both GT and Prediction are empty. Setting HD95 to 0.")
            hd95[i] = 0.0

        elif gt_empty and not pred_empty:  # Tissue is absent in ground truth
            pred_array = pred[i].cpu().numpy()  # Convert to NumPy
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
        key: float(np.nanmean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
        if key != "patient_id"
    }

    with open(filename, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(f"Saved average test set metrics to {filename}")


def ensemble_segmentation(
    test_loader,
    models_dict,
    patient_id=None,
    output_dir="./output_segmentations/simple_avg",
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
            logits_list = []
            for model_name, (model, inferer) in models_dict.items():
                logits = inferer(image).squeeze(
                    0
                )  # Remove batch dim -> (num_classes, H, W, D)
                logits_list.append(logits)

            # Average the logits across all models
            avg_logits = torch.mean(
                torch.stack(logits_list), dim=0
            )  # Shape: (num_classes, H, W, D)

            # Apply softmax and convert to segmentation map
            seg = (
                torch.nn.functional.softmax(avg_logits, dim=0)
                .argmax(dim=0)
                .unsqueeze(0)
            )  # Shape: (H, W, D)

            pred_one_hot = [(seg == i).float() for i in range(1, 4)]
            gt_one_hot = [(gt == i).float() for i in range(1, 4)]

            dice, hd95, sensitivity, specificity = compute_metrics(
                pred_one_hot, gt_one_hot
            )
            print(
                f"Dice NCR: {dice[0].item():.4f}, Dice ED: {dice[1].item():.4f}, Dice ET: {dice[2].item():.4f}\n",
                f"HD95 NCR: {hd95[0].item():.2f}, HD95 ED: {hd95[1].item():.2f}, HD95 ET: {hd95[2].item():.2f}\n",
                f"Sensitivity NCR: {sensitivity[0].item():.4f}, ED: {sensitivity[1].item():.4f}, ET: {sensitivity[2].item():.4f}\n",
                f"Specificity NCR: {specificity[0].item():.4f}, ED: {specificity[1].item():.4f}, ET: {specificity[2].item():.4f}\n",
            )

            patient_metrics.append(
                {
                    "patient_id": patient_id,
                    "Dice NCR": dice[0].item(),
                    "Dice ED": dice[1].item(),
                    "Dice ET": dice[2].item(),
                    "Dice overall": np.nanmean(dice),
                    "HD95 NCR": hd95[0].item(),
                    "HD95 ED": hd95[1].item(),
                    "HD95 ET": hd95[2].item(),
                    "HD95 overall": np.nanmean(hd95),
                    "Sensitivity NCR": sensitivity[0].item(),
                    "Sensitivity ED": sensitivity[1].item(),
                    "Sensitivity ET": sensitivity[2].item(),
                    "Sensitivity overall": np.nanmean(sensitivity),
                    "Specificity NCR": specificity[0].item(),
                    "Specificity ED": specificity[1].item(),
                    "Specificity ET": specificity[2].item(),
                    "Specificity overall": np.nanmean(specificity),
                }
            )

            seg = seg.squeeze(0)  # remove batch dimension

            # Save segmentation
            output_path = os.path.join(
                output_dir, f"simple_avg_{patient_id}_pred_seg.nii.gz"
            )
            save_segmentation_as_nifti(seg, reference_image_path, output_path)

            # Save probability maps
            probs = torch.softmax(avg_logits, dim=0)  # Shape: (4, H, W, D)
            prob_output_path = os.path.join(
                output_dir, f"simple_avg_softmax_{patient_id}.nii.gz"
            )
            save_segmentation_as_nifti(probs, reference_image_path, prob_output_path)

            torch.cuda.empty_cache()

    csv_path = os.path.join(output_dir, "simple_avg_patient_metrics.csv")
    json_path = os.path.join(output_dir, "simple_avg_average_metrics.json")
    save_metrics_csv(patient_metrics, csv_path)
    save_average_metrics(patient_metrics, json_path)



#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "01502"
    models_dict = load_all_models()
    ensemble_segmentation(config.test_loader, models_dict)
