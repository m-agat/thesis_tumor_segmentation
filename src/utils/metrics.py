import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


def compute_dice_score_per_tissue(prediction, ground_truth, tissue_label):
    """
    Compute the Dice score for a specific tissue type with single-channel prediction and ground truth.
    Args:
        prediction (numpy.ndarray): The predicted segmentation (shape: (240, 240, 155)).
        ground_truth (numpy.ndarray): The ground truth segmentation (shape: (240, 240, 155)).
        tissue_label (int): The class label of the tissue type (e.g., 1 for NCR, 2 for ED, 4 for ET).
    
    Returns:
        float: Dice score for the specified tissue type.
    """
    # Extract binary masks for the given tissue label
    gt_tissue = (ground_truth == tissue_label).astype(np.float32)
    pred_tissue = (prediction == tissue_label).astype(np.float32)
    
    intersection = np.sum(pred_tissue * gt_tissue)
    union = np.sum(pred_tissue) + np.sum(gt_tissue)
    
    if union == 0:
        return 1.0  # Perfect Dice if both are empty
    
    return (2.0 * intersection) / union


def compute_hd95(prediction, ground_truth, tissue_label):
    """
    Compute the 95th percentile Hausdorff Distance for a specific tissue type.
    Args:
        prediction (numpy.ndarray): The predicted segmentation (shape: (240, 240, 155)).
        ground_truth (numpy.ndarray): The ground truth segmentation (shape: (240, 240, 155)).
        tissue_label (int): The class label of the tissue type (e.g., 1 for NCR, 2 for ED, 4 for ET).
    
    Returns:
        float: The 95th percentile Hausdorff Distance for the specified tissue type.
    """
    # Extract binary masks for the given tissue label
    gt_tissue = (ground_truth == tissue_label).astype(np.float32)
    pred_tissue = (prediction == tissue_label).astype(np.float32)
    
    # Add batch and channel dimensions
    pred_tissue = pred_tissue[None, None, ...]
    gt_tissue = gt_tissue[None, None, ...]
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_tissue = torch.from_numpy(pred_tissue).to(device)
    gt_tissue = torch.from_numpy(gt_tissue).to(device)
    
    # Compute the 95th percentile Hausdorff Distance
    hd = compute_hausdorff_distance(
        pred_tissue, 
        gt_tissue, 
        include_background=False,
        percentile=95,
        directed=False
    )
    
    return hd.item()


def compute_metrics_with_monai(prediction, ground_truth, tissue_label, confusion_metric):
    """
    Compute sensitivity, specificity, and F1 score using MONAI's ConfusionMatrixMetric.
    Args:
        prediction (numpy.ndarray): The predicted segmentation (shape: (240, 240, 155)).
        ground_truth (numpy.ndarray): The ground truth segmentation (shape: (240, 240, 155)).
        tissue_label (int): The class label of the tissue type (e.g., 1 for NCR, 2 for ED, 4 for ET).
        confusion_metric: An instance of MONAI's ConfusionMatrixMetric.
    
    Returns:
        Tuple[float, float, float]: Sensitivity, specificity, and F1 score for the specified tissue type.
    """
    # Extract binary masks for the given tissue label
    gt_tissue = (ground_truth == tissue_label).astype(np.float32)
    pred_tissue = (prediction == tissue_label).astype(np.float32)
    
    # Add batch and channel dimensions
    pred_tissue = pred_tissue[None, None, ...]
    gt_tissue = gt_tissue[None, None, ...]
    
    # Convert to PyTorch tensors
    pred_tissue = torch.from_numpy(pred_tissue)
    gt_tissue = torch.from_numpy(gt_tissue)

    # Compute confusion metrics
    confusion_metric(y_pred=pred_tissue, y=gt_tissue)
    sensitivity, specificity, f1_score = confusion_metric.aggregate()
    confusion_metric.reset()  # Clear state for next computation
    
    return sensitivity.item(), specificity.item(), f1_score.item()


def calculate_composite_score(dice, hd95, f1_score, weights):
    """
    Calculate a composite score by combining Dice, HD95, and F1 scores.
    Args:
        dice (float): Dice score.
        hd95 (float): 95th percentile Hausdorff Distance.
        f1_score (float): F1 score.
        weights (dict): Weights for Dice, HD95, and F1 scores (keys: "Dice", "HD95", "F1").
    
    Returns:
        float: Composite score.
    """
    # Handle HD95 if it's NaN or infinite
    if np.isnan(hd95) or np.isinf(hd95):
        hd95_score = 0  # Default to 0 if HD95 is not computable
    else:
        hd95_score = 1 / (1 + hd95)

    # Set F1 to 0 if it's NaN
    if np.isnan(f1_score):
        f1_score = 0.0

    # Calculate the weighted sum, handling cases where dice might be NaN
    dice = dice if not np.isnan(dice) else 0.0
    return (weights["Dice"] * dice +
            weights["HD95"] * hd95_score +
            weights["F1"] * f1_score)