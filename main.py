import os
import torch
import nibabel as nib
from dataset.dataloaders import load_test_data
from src.model_inference import load_models, get_ensemble_segmentation_with_uncertainty, compare_predictions_to_ground_truth
from src.utils import create_case_folder, save_segmentation_as_nifti
from src.visualization import plot_segmentation_with_accuracy_and_uncertainty

def process_case(case_num, data_dir, results_dir, models, device):
    """
    Process a single case and generate the ensemble segmentation, uncertainty, and visualizations.
    """
    test_loader = load_test_data(data_dir, case_num)

    # Load nnUNet segmentation result and add it to the models dictionary
    nnunet_segmentation_path = os.path.join(results_dir, f"segmentations/BraTS2021_{case_num}/BraTS2021_{case_num}_nnunet_segmentation.nii.gz")
    if not os.path.exists(nnunet_segmentation_path):
        raise FileNotFoundError(f"nnUNet segmentation result not found at: {nnunet_segmentation_path}")
    nnunet_segmentation_result = nib.load(nnunet_segmentation_path).get_fdata()

    # Treat nnUNet as a "model" in the models dictionary
    models['nnUNet'] = nnunet_segmentation_result

    # Load ground truth segmentation
    ground_truth_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_seg.nii.gz")
    ground_truth = nib.load(ground_truth_path).get_fdata()

    # Call the ensemble segmentation function
    segmentation, variance_uncertainty, dice_uncertainty = get_ensemble_segmentation_with_uncertainty(
        models, test_loader, roi=(96, 96, 96), ground_truth=ground_truth
    )

    # Compare the predicted segmentation with the ground truth
    accuracy_map = compare_predictions_to_ground_truth(segmentation, ground_truth)

    # Load the original image for visualization
    img_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t1ce.nii.gz")
    img = nib.load(img_path).get_fdata()

    # Define the slice number for visualization
    slice_num = 68

    # Plot segmentation, accuracy, and uncertainty maps
    plot_segmentation_with_accuracy_and_uncertainty(
        img, ground_truth, segmentation, accuracy_map, variance_uncertainty, dice_uncertainty, slice_num, 
        save_path=f"./figures/{case_num}_segmentation_plot.png"
    )

    # Save the final ensemble segmentation
    case_folder = create_case_folder(results_dir, case_num)
    save_segmentation_as_nifti(segmentation, img_path, os.path.join(case_folder, f"{case_num}_ensemble_segmentation.nii.gz"))


def main():
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data", "brats2021challenge", "TrainingData")
    results_dir = os.path.join(root_dir, "results")

    case_dirs = [case for case in os.listdir(data_dir) if case.startswith("BraTS2021_")]
    
    # Extract the case numbers from the directory names (e.g., BraTS2021_00000 -> 00000)
    case_nums = [case.split('_')[-1] for case in case_dirs]
    
    case_nums = sorted(case_nums)
    case_nums = case_nums[:10]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, results_dir)

    for case_num in case_nums:
        print(f"Processing case: BraTS2021_{case_num}")
        try:
            process_case(case_num, data_dir, results_dir, models, device)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing case {case_num}: {e}")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
