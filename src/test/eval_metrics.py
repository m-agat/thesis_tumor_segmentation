from typing import Tuple
from medpy import metric
import numpy as np

def get_confusion_matrix(prediction: np.ndarray, reference: np.ndarray, roi_mask: np.ndarray = None) -> Tuple[int, int, int, int]:
    """
    Computes tp, fp, tn, fn from the provided segmentations.
    If roi_mask is provided, only voxels within this region of interest are considered.
    """
    assert prediction.shape == reference.shape, "'prediction' and 'reference' must have the same shape"

    if roi_mask is None:
        roi_mask = np.ones_like(prediction)

    tp = int((roi_mask * (prediction != 0) * (reference != 0)).sum())  # True Positives
    fp = int((roi_mask * (prediction != 0) * (reference == 0)).sum())  # False Positives
    tn = int((roi_mask * (prediction == 0) * (reference == 0)).sum())  # True Negatives
    fn = int((roi_mask * (prediction == 0) * (reference != 0)).sum())  # False Negatives

    return tp, fp, tn, fn

# Dice computation using TP, FP, FN
def dice(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0 if denominator == 0 else (2 * tp / denominator)

# Hausdorff Distance (HD95) using medpy
def hausdorff_95(prediction: np.ndarray, reference: np.ndarray) -> float:
    try:
        return metric.hd95(prediction, reference)
    except Exception as e:
        print("Error:", e)
        print(f"Prediction labels {np.unique(prediction)}, Ground Truth labels {np.unique(reference)}")
        return 373  # Large value to signify an error

# Sensitivity (Recall)
def recall(tp: int, fn: int) -> float:
    actual_positives = tp + fn
    return 0 if actual_positives == 0 else (tp / actual_positives)

# Specificity (Precision)
def precision(tp: int, fp: int) -> float:
    predicted_positives = tp + fp
    return 0 if predicted_positives == 0 else (tp / predicted_positives)

# Updated calculate_metrics function
def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Calculate Dice, HD95, Sensitivity, and Specificity for WT, TC, ET subregions.
    """
    metrics = {}

    # Define masks for WT, TC, ET
    wt_pred = (pred > 0).astype(np.int8)  # WT combines all tumor regions
    wt_gt = (gt > 0).astype(np.int8)
    
    tc_pred = ((pred == 1) | (pred == 4)).astype(np.int8)  # TC: NCR/NET + ET
    tc_gt = ((gt == 1) | (gt == 4)).astype(np.int8)
    
    et_pred = (pred == 4).astype(np.int8)  # ET: label 4
    et_gt = (gt == 4).astype(np.int8)

    # Compute metrics for each subregion
    for region, (p, g) in zip(["WT", "TC", "ET"], [(wt_pred, wt_gt), (tc_pred, tc_gt), (et_pred, et_gt)]):
        tp, fp, tn, fn = get_confusion_matrix(p, g)
        metrics[f"{region}_Dice"] = dice(tp, fp, fn)
        # metrics[f"{region}_HD95"] = hausdorff_95(p, g)
        metrics[f"{region}_Sensitivity"] = recall(tp, fn)
        metrics[f"{region}_Specificity"] = precision(tp, fp)

    return metrics
