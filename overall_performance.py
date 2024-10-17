import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from dataset.dataloaders_binary import load_test_data
from src.model_inference import load_models, get_segmentation
from src.metrics import compute_dice_score


def process_case(case_num, data_dir, results_dir, models, device):
    """
    Process a single case and compute the overall Dice score for each model.
    """
    torch.cuda.empty_cache()
    test_loader = load_test_data(data_dir, case_num)

    nnunet_segmentation_path = os.path.join(results_dir, f"segmentations/BraTS2021_{case_num}/BraTS2021_{case_num}_nnunet_segmentation.nii.gz")
    if not os.path.exists(nnunet_segmentation_path):
        raise FileNotFoundError(f"nnUNet segmentation result not found at: {nnunet_segmentation_path}")
    nnunet_segmentation_result = nib.load(nnunet_segmentation_path).get_fdata()

    ground_truth_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_seg.nii.gz")
    ground_truth = nib.load(ground_truth_path).get_fdata()

    # Dictionary to store overall Dice scores for each model
    dice_scores = {model_name: 0.0 for model_name in models.keys()}
    dice_scores['nnUNet'] = 0.0  # Add nnUNet to dice_scores

    roi = (96, 96, 96)
    
    # Process each model individually
    for model_name, model in models.items():
        segmentation = get_segmentation(model, roi, test_loader, overlap=0.6, device=device)

        # Compute overall Dice score for the entire segmentation
        dice_scores[model_name] = compute_dice_score(segmentation, ground_truth)

    # Compute Dice score for nnUNet
    dice_scores['nnUNet'] = compute_dice_score(nnunet_segmentation_result, ground_truth)

    print(f"Dice scores for case {case_num}: {dice_scores}")

    return dice_scores


def main():
    # Paths
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data", "brats2021challenge", "TrainingData")
    results_dir = os.path.join(root_dir, "results")
    model_comparison_dir = os.path.join(results_dir, "model_comparison")
    os.makedirs(model_comparison_dir, exist_ok=True)

    case_dirs = [case for case in os.listdir(data_dir) if case.startswith("BraTS2021_")]
    
    # Extract the case numbers from the directory names (e.g., BraTS2021_00000 -> 00000)
    case_nums = [case.split('_')[-1] for case in case_dirs]
    case_nums = sorted(case_nums)
    case_nums = case_nums[:30]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, results_dir)

    # Initialize a dictionary to accumulate Dice scores across cases for each model
    overall_dice_scores = {model_name: [] for model_name in models.keys()}
    overall_dice_scores['nnUNet'] = []

    # Process each case in the list
    for case_num in case_nums:
        print(f"Processing case: BraTS2021_{case_num}")
        try:
            dice_scores = process_case(case_num, data_dir, results_dir, models, device)
            
            # Accumulate Dice scores across cases
            for model_name, score in dice_scores.items():
                overall_dice_scores[model_name].append(score)

        except Exception as e:
            print(f"Error processing case {case_num}: {e}")

    # Compute the average Dice score across all cases for each model
    final_dice_scores = {model_name: np.mean(scores) for model_name, scores in overall_dice_scores.items()}

    # Convert results to pandas DataFrame for saving
    results = []
    for model_name, avg_dice_score in final_dice_scores.items():
        results.append({
            'Model': model_name,
            'Average Dice Score': avg_dice_score
        })

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(model_comparison_dir, 'overall_dice_scores.csv')
    results_df.to_csv(csv_path, index=False)

    print(f"Overall Dice scores saved to {csv_path}")


if __name__ == "__main__":
    main()
