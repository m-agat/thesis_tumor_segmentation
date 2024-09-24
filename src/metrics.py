import numpy as np


def compute_dice_score(prediction, ground_truth, tissue_type):
    """
    Compute the Dice score for a specific tissue type.
    Args:
        prediction (numpy.ndarray): The predicted segmentation.
        ground_truth (numpy.ndarray): The ground truth segmentation.
        tissue_type (int): The label of the tissue type to evaluate (e.g., 1 for NCR, 2 for ED, 3 for ET).
    
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