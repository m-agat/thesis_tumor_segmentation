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

#####################
#### Load Models ####
#####################

REGIONS = ["BG", "NCR", "ED", "ET"]
DEVICE = config.device
ROI = config.roi
SW_BATCH = config.sw_batch_size
OVERLAP = config.infer_overlap

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

def compute_metrics(pred_list, gt_list):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    pred_list, gt_list: lists of length C, each tensor [H,W,D]
    """
    # Prepare batch-channel tensors
    y_pred = torch.stack([torch.as_tensor(p) for p in pred_list], dim=0).unsqueeze(0)
    y_gt   = torch.stack([torch.as_tensor(g) for g in gt_list],   dim=0).unsqueeze(0)

    # Dice (including background)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.NONE,
        get_not_nans=True,
    )
    dice_metric(y_pred=y_pred, y=y_gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.squeeze(0).cpu().numpy()  # shape: (C,)
    not_nans     = not_nans.squeeze(0).cpu().numpy()    # shape: (C,)

    # Correct Dice when GT empty
    for i in range(len(dice_scores)):
        if not_nans[i] == 0:
            dice_scores[i] = 1.0 if pred_list[i].sum().item() == 0 else 0.0

    hd95 = np.zeros(len(dice_scores), dtype=float)
    for i in range(len(pred_list)):
        # prepare single-channel batch as [1,1,H,W,D]
        pred_ch = pred_list[i].unsqueeze(0).unsqueeze(0)
        gt_ch   = gt_list[i].unsqueeze(0).unsqueeze(0)
        pred_empty = torch.sum(pred_ch).item() == 0
        gt_empty   = not_nans[i] == 0
        if pred_empty and gt_empty:
            # both empty
            hd95[i] = 0.0
        elif gt_empty and not pred_empty:
            # GT absent, prediction present
            pred_array = pred_ch.cpu().numpy()[0,0]
            if pred_array.sum() > 0:
                com = center_of_mass(pred_array)
                if any(np.isnan(com)):
                    hd95[i] = 0.0
                else:
                    com_mask = np.zeros_like(pred_array, dtype=np.uint8)
                    coords = tuple(map(int, map(round, com)))
                    com_mask[coords] = 1
                    mask_tensor = torch.from_numpy(com_mask).unsqueeze(0).unsqueeze(0).to(torch.float32).to(DEVICE)
                    mock_hd = compute_hausdorff_distance(
                        y_pred=pred_ch,
                        y=mask_tensor,
                        include_background=False,
                        distance_metric="euclidean",
                        percentile=95,
                    )
                    hd95[i] = float(mock_hd.squeeze().item())
            else:
                hd95[i] = 0.0
        elif pred_empty and not gt_empty:
            # Prediction empty, GT present
            gt_array = gt_ch.cpu().numpy()[0,0]
            if gt_array.sum() > 0:
                com = center_of_mass(gt_array)
                if any(np.isnan(com)):
                    hd95[i] = 0.0
                else:
                    com_mask = np.zeros_like(gt_array, dtype=np.uint8)
                    coords = tuple(map(int, map(round, com)))
                    com_mask[coords] = 1
                    mask_tensor = torch.from_numpy(com_mask).unsqueeze(0).unsqueeze(0).to(torch.float32).to(DEVICE)
                    mock_hd = compute_hausdorff_distance(
                        y_pred=gt_ch,
                        y=mask_tensor,
                        include_background=False,
                        distance_metric="euclidean",
                        percentile=95,
                    )
                    hd95[i] = float(mock_hd.squeeze().item())
            else:
                hd95[i] = 0.0
        else:
            # both present: direct hd95
            hd_t = compute_hausdorff_distance(
                y_pred=pred_ch,
                y=gt_ch,
                include_background=False,
                distance_metric="euclidean",
                percentile=95,
            )
            hd95[i] = float(hd_t.squeeze().item())

    # Sensitivity / Specificity
    conf_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name=["sensitivity","specificity"],
        reduction="none",
    )
    conf_metric(y_pred=y_pred, y=y_gt)
    sens, spec = conf_metric.aggregate()
    sens = sens.squeeze(0).cpu().numpy()
    spec = spec.squeeze(0).cpu().numpy()

    # Correct sens/spec when GT empty
    for i in range(len(sens)):
        if not_nans[i] == 0:
            sens[i] = 1.0 if pred_list[i].sum().item() == 0 else 0.0
            spec[i] = 1.0

    return dice_scores, hd95, sens, spec


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
    ood=False
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
            if ood:
                patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(test_loader.dataset[0]["path"]))))
            else:
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
                logits = inferer(image).squeeze(0)  # Remove batch dim -> (num_classes, H, W, D)
                logits_list.append(logits)

            # Average the logits across all models
            avg_logits = torch.mean(
                torch.stack(logits_list), dim=0
            )  # Shape: (num_classes, H, W, D)

            # Apply softmax and convert to segmentation map
            seg = torch.nn.functional.softmax(avg_logits, dim=0).argmax(dim=0) # Shape: (H, W, D)

            pred_list = [(seg == i).float() for i in range(len(REGIONS))]
            if gt.shape[1] == len(REGIONS):
                gt_list = [gt[:, i].squeeze(0) for i in range(len(REGIONS))]
            else:
                gt_list = [(gt == i).float().squeeze(0) for i in range(len(REGIONS))]

            # Compute performance metrics
            dice_scores, hd95, sens, spec = compute_metrics(pred_list, gt_list)
            patient_metrics.append({
                'patient_id': patient_id,
                **{f"Dice {REGIONS[i]}": dice_scores[i] for i in range(1, len(REGIONS))},
                **{f"HD95 {REGIONS[i]}": hd95[i] for i in range(1, len(REGIONS))},
                **{f"Sensitivity {REGIONS[i]}": sens[i] for i in range(1, len(REGIONS))},
                **{f"Specificity {REGIONS[i]}": spec[i] for i in range(1, len(REGIONS))},
                'Dice overall': float(np.mean(dice_scores[1:])),
                'HD95 overall': float(np.mean(hd95[1:])),
                'Sensitivity overall': float(np.mean(sens[1:])),
                'Specificity overall': float(np.mean(spec[1:])),
            })
            
            print("--- Simple averaging results ---")
            print("Dice: ", dice_scores)
            print("HD95: ", hd95)
            print("Sensitivity: ", sens)
            print("Specificity: ", spec)

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

    # csv_path = os.path.join(output_dir, "simple_avg_patient_metrics.csv")
    # json_path = os.path.join(output_dir, "simple_avg_average_metrics.json")
    # save_metrics_csv(patient_metrics, csv_path)
    # save_average_metrics(patient_metrics, json_path)



#######################
#### Run Inference ####
#######################

# if __name__ == "__main__":
#     patient_id = "01502"
#     models_dict = load_all_models()
#     ensemble_segmentation(config.test_loader, models_dict)
