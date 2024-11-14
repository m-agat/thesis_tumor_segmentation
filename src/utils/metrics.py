import numpy as np 
import scipy.spatial.distance as distance
from monai.metrics import compute_hausdorff_distance
import torch 

def compute_dice_score_per_tissue(prediction, ground_truth, tissue_type):
    """
    Compute the Dice score for a specific tissue type.
    Args:
        prediction (numpy.ndarray): The predicted segmentation.
        ground_truth (numpy.ndarray): The ground truth segmentation.
        tissue_type (int): The label of the tissue type to evaluate (e.g., 1 for NCR, 2 for ED, 4 for ET).
    
    Returns:
        float: Dice score for the specified tissue type.
    """
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)
    
    intersection = np.sum(pred_tissue * gt_tissue)
    union = np.sum(pred_tissue) + np.sum(gt_tissue)
    
    if union == 0:
        return 1.0  # If both prediction and ground truth have no pixels for this class, Dice is perfect.
    
    return (2.0 * intersection) / union


def compute_hd95(prediction, ground_truth, tissue_type):
    # Convert binary masks for the specified tissue type
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)
    
    # Add batch and channel dimensions
    if pred_tissue.ndim == 3:
        pred_tissue = pred_tissue[None, None, ...]  # Shape: [1, 1, H, W, D]
    if gt_tissue.ndim == 3:
        gt_tissue = gt_tissue[None, None, ...]      # Shape: [1, 1, H, W, D]
    
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

def compute_metrics_with_monai(prediction, ground_truth, tissue_type, confusion_metric):
    # Convert binary masks for the specific tissue type
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)

    # Add batch and channel dimensions if necessary
    if pred_tissue.ndim == 3:
        pred_tissue = pred_tissue[None, None, ...]  # Shape: [1, 1, H, W, D]
    if gt_tissue.ndim == 3:
        gt_tissue = gt_tissue[None, None, ...]      # Shape: [1, 1, H, W, D]
    
    # Convert to PyTorch tensors
    pred_tissue = torch.from_numpy(pred_tissue)
    gt_tissue = torch.from_numpy(gt_tissue)

    # Ensure binary format (one-hot encoding if needed)
    confusion_metric(y_pred=pred_tissue, y=gt_tissue)

    # Aggregate metrics and retrieve results
    sensitivity, specificity, f1_score = confusion_metric.aggregate()
    
    # Clear state for the next sample (important for batch processing)
    confusion_metric.reset()
    
    return sensitivity.item(), specificity.item(), f1_score.item()


def calculate_composite_score(dice, hd95, f1_score, weights):
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