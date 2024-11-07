import numpy as np
import pandas as pd 


def compute_dice_score(prediction, ground_truth):
    """
    Calculate Dice score for the given prediction and ground truth segmentation.
    This can be used as a measure of uncertainty.
    """
    intersection = np.sum((prediction == ground_truth) & (ground_truth > 0))
    total = np.sum(prediction > 0) + np.sum(ground_truth > 0)

    if total == 0:
        return 1.0  # If both prediction and ground truth have no foreground, Dice is perfect
    
    return (2.0 * intersection) / total

def compute_dice_score_per_tissue(prediction, ground_truth, tissue_type):
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

def compute_model_weights(dice_scores):
    """
    Compute weights for each model based on their Dice scores.
    Models with higher Dice scores receive higher weights.
    """
    total_score = np.sum(dice_scores)
    if total_score == 0:
        return np.ones(len(dice_scores)) / len(dice_scores)  # If no dice score, equal weights
    return dice_scores / total_score  # Normalize so that weights sum to 1


def load_dice_scores(csv_path):
    """
    Load dice scores from a CSV file and return them as normalized weights.
    The weights will sum up to 1.
    """
    dice_scores_df = pd.read_csv(csv_path)
    
    dice_scores = dict(zip(dice_scores_df['Model'], dice_scores_df['Average Dice Score']))
    
    total_score = sum(dice_scores.values())
    weights = {model: score / total_score for model, score in dice_scores.items()}
    
    return weights

def load_per_tissue_dice_scores(csv_path):
    """
    Load per-tissue dice scores from a CSV file and return them as normalized exponential weights
    for each tissue type.
    """
    dice_scores_df = pd.read_csv(csv_path)
    
    tissues = ['Background', 'NCR', 'ED', 'ET']
    weights = {tissue: {} for tissue in tissues}
    
    for tissue in tissues:
        exp_dice_scores = np.exp(dice_scores_df[f'{tissue} Dice']) # Exponential Dice Scores
        total_exp_score = exp_dice_scores.sum()
        scaling_factor = 4
        for _, row in dice_scores_df.iterrows():
            model = row['Model']
            weights[tissue][model] = np.exp(row[f'{tissue} Dice'] * scaling_factor) / total_exp_score # Normalized Dice scores
    
    return weights


def calculate_variance_based_certainty(predictions):
    """
    Calculate certainty for each voxel based on the variance of predictions across models.
    
    Parameters:
    - predictions: Numpy array of predictions from different models, shape [num_models, H, W, D]
    
    Returns:
    - certainty: Certainty score for each voxel (1 - variance)
    """
    # Calculate variance across the model predictions for each voxel
    variance_map = np.var(predictions, axis=0)  # Variance across models at each voxel
    
    # Certainty is inversely related to variance (lower variance = higher certainty)
    certainty = 1 - variance_map  # Certainty: 1 means high certainty, 0 means low certainty
    
    return certainty