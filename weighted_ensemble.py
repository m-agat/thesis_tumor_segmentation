import os
import torch
import nibabel as nib
from src.data_loading import load_test_data
from src.model_inference import load_models, get_weighted_ensemble_segmentation_with_uncertainty, compare_predictions_to_ground_truth
from src.utils import create_case_folder, save_segmentation_as_nifti
from src.visualization import plot_segmentation_with_accuracy_and_uncertainty
from src.utils import csv_to_dict

def process_case(case_num, data_dir, results_dir, models, model_dice_scores, device):
    """
    Process a single case and generate the ensemble segmentation, uncertainty, and visualizations.
    Includes nnUNet as part of the weighted ensemble.
    """
    test_loader = load_test_data(data_dir, case_num)

    # Load nnUNet segmentation result and add it to the models dictionary as if it's another model
    nnunet_segmentation_path = os.path.join(results_dir, f"segmentations/BraTS2021_{case_num}/BraTS2021_{case_num}_nnunet_segmentation.nii.gz")
    if not os.path.exists(nnunet_segmentation_path):
        raise FileNotFoundError(f"nnUNet segmentation result not found at: {nnunet_segmentation_path}")
    nnunet_segmentation_result = nib.load(nnunet_segmentation_path).get_fdata()

    print(f"nnUNet output shape: {nnunet_segmentation_result.shape}")

    # Add nnUNet result as a model in the dictionary, treating it like the others
    models['nnUNet'] = nnunet_segmentation_result

    # Load ground truth segmentation
    ground_truth_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_seg.nii.gz")
    ground_truth = nib.load(ground_truth_path).get_fdata()

    print(f"Ground truth shape: {ground_truth.shape}")

    # Print models and their Dice scores
    print("Models and their corresponding Dice scores:")
    for model_name in models.keys():
        if model_name in model_dice_scores:
            print(f"Model: {model_name}, Dice Score: {model_dice_scores[model_name]}")
        else:
            print(f"Warning: No Dice score found for model: {model_name}")

    # Perform weighted ensemble segmentation
    segmentation, variance_uncertainty, dice_uncertainty = get_weighted_ensemble_segmentation_with_uncertainty(
        models, test_loader, roi=(96, 96, 96), ground_truth=ground_truth, model_dice_scores=model_dice_scores
    )

    # Compare predicted segmentation to ground truth
    accuracy_map = compare_predictions_to_ground_truth(segmentation, ground_truth)

    # Load the original image for visualization
    img_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t1ce.nii.gz")
    img = nib.load(img_path).get_fdata()

    # Define the slice number for 2D visualization
    slice_num = 68

    # Plot segmentation, accuracy, and uncertainty maps
    plot_segmentation_with_accuracy_and_uncertainty(
        img, ground_truth, segmentation, accuracy_map, variance_uncertainty, dice_uncertainty, slice_num, 
        save_path=f"./figures/{case_num}_weighted_segmentation_plot.png"
    )

    # Save the final ensemble segmentation
    case_folder = create_case_folder(results_dir, case_num)
    save_segmentation_as_nifti(segmentation, img_path, os.path.join(case_folder, f"{case_num}_weighted_ensemble_segmentation.nii.gz"))


def main():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data", "brats2021challenge", "TrainingData")
    results_dir = os.path.join(root_dir, "results")
    model_dice_scores_path = "./results/model_comparison/overall_dice_scores.csv"
    model_dice_scores = csv_to_dict(model_dice_scores_path, key_column='Model', value_column='Average Dice Score')

    # Get the list of all case directories in the TrainingData folder
    case_dirs = [case for case in os.listdir(data_dir) if case.startswith("BraTS2021_")]
    
    # Extract the case numbers from the directory names (e.g., BraTS2021_00000 -> 00000)
    case_nums = [case.split('_')[-1] for case in case_dirs]
    
    case_nums = sorted(case_nums)
    case_nums = case_nums[:1]

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, results_dir)

    # Process each case
    for case_num in case_nums:
        print(f"Processing case: BraTS2021_{case_num}")
        try:
            process_case(case_num, data_dir, results_dir, models, model_dice_scores, device)
            torch.cuda.empty_cache()  # Clear GPU memory after each case
        except Exception as e:
            print(f"Error processing case {case_num}: {e}")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()