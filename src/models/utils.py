
import numpy as np
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
import re
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import pandas as pd
import torch 
import src.config.config as config

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


def extract_patient_id(path):
    """
    Extract patient ID from the file path.
    """
    numbers = re.findall(r"\d+", path)
    patient_id = numbers[-1]
    return patient_id