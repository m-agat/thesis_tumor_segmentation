import pandas as pd
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import re
from scipy.stats import spearmanr

# Base paths for ground truth and predicted outputs
gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
pred_base_path = "../ensemble/output_segmentations/ttd/"

# List image files
gt_images = sorted(os.listdir(gt_base_path))
pred_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'ttd.*_pred_seg.nii.gz', f)])
prob_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'ttd_softmax_.*.nii.gz', f)])
uncertainty_NCR_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_NCR.*_fused.nii.gz', f)])
uncertainty_ED_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ED.*_fused.nii.gz', f)])
uncertainty_ET_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ET.*_fused.nii.gz', f)])

# Extract patient IDs from prediction filenames
pred_patient_ids = [re.search(r'ttd_(\d{5})_pred_seg', f).group(1) for f in pred_images]
# Filter gt_images to only include matching patient IDs
gt_images = sorted([f for f in os.listdir(gt_base_path) 
                    if any(pid in f for pid in pred_patient_ids)])

def compute_expected_calibration_error(probabilities, gt, num_bins=10, subregion=None):
    """
    Compute the Expected Calibration Error (ECE) for a specific subregion or overall.

    Args:
        probabilities (np.array): Softmax probabilities of shape (C, H, W, D)
                                  where C is the number of classes.
        gt (np.array): Ground truth labels of shape (H, W, D) with integer class labels.
        num_bins (int): Number of bins to use for calibration computation.
        subregion (int, optional): Specific subregion to analyze (1 for NCR, 2 for ED, 4 for ET).
                                 If None, analyzes all regions.

    Returns:
        float: The computed ECE value.
    """
    # Compute predicted labels and their confidences (max probability per voxel)
    predicted_labels = np.argmax(probabilities, axis=0)  # Shape: (H, W, D)
    confidences = np.max(probabilities, axis=0)            # Shape: (H, W, D)

    # Create a binary correctness map: 1 if prediction is correct, 0 otherwise.
    correctness = (predicted_labels == gt).astype(np.float32)

    # If analyzing a specific subregion, mask out other regions
    if subregion is not None:
        mask = (gt == subregion)
        confidences = confidences[mask]
        correctness = correctness[mask]
        if len(confidences) == 0:  # If no voxels in this subregion
            return 0.0

    # Flatten to work with all voxels at once.
    confidences_flat = confidences.flatten()
    correctness_flat = correctness.flatten()
    total_voxels = len(confidences_flat)

    if total_voxels == 0:
        return 0.0

    ece = 0.0
    # Create equally spaced bins [0, 1]
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        # Find voxels within the current bin
        in_bin = (confidences_flat >= bin_lower) & (confidences_flat < bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_confidence = np.mean(confidences_flat[in_bin])
            avg_accuracy = np.mean(correctness_flat[in_bin])
            ece += (bin_count / total_voxels) * np.abs(avg_confidence - avg_accuracy)
    return ece

# List for storing the ECE values per patient and per subregion
ece_values = []

# Define subregions (NCR=1, ED=2, ET=3)
subregions = {
    'NCR': 1,
    'ED': 2,
    'ET': 3
}

# Loop through the images (each patient)
for gt_image, pred_image, prob_image, uncertainty_ncr_image, uncertainty_ed_image, uncertainty_et_image in zip(
    gt_images, pred_images, prob_images, uncertainty_NCR_images, uncertainty_ED_images, uncertainty_ET_images
):
    # Construct full paths for the ground truth and predictions
    gt_image_folder = rf"{gt_image}\{gt_image}_seg.nii.gz"
    gt_image_path = os.path.join(gt_base_path, gt_image_folder)
    pred_image_path = os.path.join(pred_base_path, pred_image)
    prob_image_path = os.path.join(pred_base_path, prob_image)
    uncertainty_ncr_path = os.path.join(pred_base_path, uncertainty_ncr_image)
    uncertainty_ed_path = os.path.join(pred_base_path, uncertainty_ed_image)
    uncertainty_et_path = os.path.join(pred_base_path, uncertainty_et_image)

    # Load the images using nibabel
    gt_image_nii = nib.load(gt_image_path)
    gt_image_data = gt_image_nii.get_fdata()

    pred_image_nii = nib.load(pred_image_path)
    pred_image_data = pred_image_nii.get_fdata()

    prob_image_nii = nib.load(prob_image_path)
    softmax_prob = prob_image_nii.get_fdata()

    # Compute overall ECE
    overall_ece = compute_expected_calibration_error(softmax_prob, gt_image_data, num_bins=10)
    
    # Compute ECE for each subregion
    subregion_eces = {}
    for subregion_name, subregion_value in subregions.items():
        subregion_ece = compute_expected_calibration_error(softmax_prob, gt_image_data, num_bins=10, subregion=subregion_value)
        subregion_eces[subregion_name] = subregion_ece

    # Store results
    ece_values.append({
        "patient": gt_image,
        "Overall_ECE": overall_ece,
        "NCR_ECE": subregion_eces['NCR'],
        "ED_ECE": subregion_eces['ED'],
        "ET_ECE": subregion_eces['ET']
    })

    print(f"Patient {gt_image}:")
    print(f"  Overall ECE: {overall_ece:.4f}")
    for subregion_name, ece in subregion_eces.items():
        print(f"  {subregion_name} ECE: {ece:.4f}")

# Create a DataFrame and save the ECE values to a CSV file
ece_df = pd.DataFrame(ece_values)
ece_csv_path = os.path.join(pred_base_path, "ece_results_per_subregion.csv")
ece_df.to_csv(ece_csv_path, index=False)
print(f"Saved ECE results to {ece_csv_path}")
