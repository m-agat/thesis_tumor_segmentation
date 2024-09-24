import os
import torch
import nibabel as nib
from src.data_loading import load_test_data
from src.model_inference import load_models, get_ensemble_segmentation_with_uncertainty, compare_predictions_to_ground_truth
from src.utils import create_case_folder, save_segmentation_as_nifti
from src.visualization import plot_segmentation_with_accuracy_and_uncertainty

def process_case(case_num, data_dir, results_dir, models, device):
    """
    Process a single case and generate the ensemble segmentation, uncertainty, and visualizations.
    """
    # Load data (test_loader contains both image and label)
    test_loader = load_test_data(data_dir, case_num)

    # Load nnUNet segmentation result for comparison
    nnunet_segmentation_path = os.path.join(results_dir, f"segmentations/BraTS2021_{case_num}/BraTS2021_{case_num}_nnunet_segmentation.nii.gz")
    if not os.path.exists(nnunet_segmentation_path):
        raise FileNotFoundError(f"nnUNet segmentation result not found at: {nnunet_segmentation_path}")
    nnunet_segmentation_result = nib.load(nnunet_segmentation_path).get_fdata()

    # Load ground truth segmentation
    ground_truth_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_seg.nii.gz")
    ground_truth = nib.load(ground_truth_path).get_fdata()

    # Get ensemble segmentation and uncertainty
    segmentation, adjusted_uncertainty = get_ensemble_segmentation_with_uncertainty(
        models, test_loader, roi=(96, 96, 96), ground_truth=ground_truth, nnunet_segmentation_result=nnunet_segmentation_result
    )

    # Compare predicted segmentation with ground truth to get accuracy map
    accuracy_map = compare_predictions_to_ground_truth(segmentation, ground_truth)

    # Load image (for visualization purposes)
    img_path = os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t1ce.nii.gz")
    img = nib.load(img_path).get_fdata()

    # Define slice number for 2D visualization
    slice_num = 68

    # Visualize the original image, ground truth, segmentation, accuracy map, and uncertainty map
    plot_segmentation_with_accuracy_and_uncertainty(
        img, ground_truth, segmentation, accuracy_map, adjusted_uncertainty, slice_num, 
        save_path=f"./figures/{case_num}_segmentation_plot.png"
    )

    # Save the final ensemble segmentation as a NIfTI file
    case_folder = create_case_folder(results_dir, case_num)
    save_segmentation_as_nifti(segmentation, img_path, os.path.join(case_folder, f"{case_num}_ensemble_segmentation.nii.gz"))

def main():
    # Paths
    root_dir = os.getcwd()
    data_dir = os.path.join(root_dir, "data", "brats2021challenge", "TrainingData")
    results_dir = os.path.join(root_dir, "results")

    # Get the list of all case directories in the TrainingData folder
    case_dirs = [case for case in os.listdir(data_dir) if case.startswith("BraTS2021_")]
    
    # Extract the case numbers from the directory names (e.g., BraTS2021_00000 -> 00000)
    case_nums = [case.split('_')[-1] for case in case_dirs]
    
    # Sort the cases (optional, to ensure they are processed in order)
    case_nums = sorted(case_nums)

    # Limit to 20 cases (optional, based on your requirement)
    case_nums = case_nums[:20]

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device, results_dir)

    # Process each case in the list
    for case_num in case_nums:
        print(f"Processing case: BraTS2021_{case_num}")
        try:
            process_case(case_num, data_dir, results_dir, models, device)
        except Exception as e:
            print(f"Error processing case {case_num}: {e}")

if __name__ == "__main__":
    main()
