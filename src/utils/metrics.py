import numpy as np 
import scipy.spatial.distance as distance

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


def compute_hd95(prediction, ground_truth, tissue_type, batch_size=10000):
    """
    Compute the 95th percentile Hausdorff Distance for a specific tissue type.
    Uses batch processing to handle memory constraints.
    """
    pred_tissue = (prediction == tissue_type).astype(np.uint8)
    gt_tissue = (ground_truth == tissue_type).astype(np.uint8)

    if pred_tissue.ndim != 3 or gt_tissue.ndim != 3:
        raise ValueError("Expected 3D arrays for prediction and ground truth.")

    # Get the surface points
    pred_points = np.argwhere(pred_tissue)
    gt_points = np.argwhere(gt_tissue)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')  # No prediction or ground truth points

    # Function to compute minimum distances in batches
    def batch_min_distances(points_a, points_b, batch_size):
        min_distances = []
        for i in range(0, len(points_a), batch_size):
            batch_a = points_a[i:i + batch_size]
            dist_batch = distance.cdist(batch_a, points_b)
            min_distances.extend(dist_batch.min(axis=1))  # Store minimum distance per point in batch
        return np.array(min_distances)

    # Compute 95th percentile distances with batch processing
    distances_pred_to_gt = batch_min_distances(pred_points, gt_points, batch_size)
    distances_gt_to_pred = batch_min_distances(gt_points, pred_points, batch_size)

    hd95_pred_to_gt = np.percentile(distances_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(distances_gt_to_pred, 95)

    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def compute_sensitivity(prediction, ground_truth, tissue_type):
    """
    Compute the Sensitivity (Recall) for a specific tissue type.
    """
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)
    
    true_positive = np.sum(pred_tissue * gt_tissue)
    false_negative = np.sum(gt_tissue * (1 - pred_tissue))
    
    if true_positive + false_negative == 0:
        return 1.0  # Perfect score if no relevant tissue exists
    
    return true_positive / (true_positive + false_negative)

def compute_specificity(prediction, ground_truth, tissue_type):
    """
    Compute the Specificity for a specific tissue type.
    """
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)
    
    true_negative = np.sum((1 - pred_tissue) * (1 - gt_tissue))
    false_positive = np.sum(pred_tissue * (1 - gt_tissue))
    
    if true_negative + false_positive == 0:
        return 1.0  # Perfect score if no relevant tissue exists
    
    return true_negative / (true_negative + false_positive)


def calculate_composite_score(dice, hd95, sensitivity, weights):
    # Normalize hd95 (lower is better)
    hd95 = 1 / (1 + hd95)  # Transform to a 0-1 range where lower HD95 gives higher score

    # Calculate the weighted sum
    return (weights["Dice"] * dice +
            weights["HD95"] * hd95 +
            weights["Sensitivity"] * sensitivity 
            )