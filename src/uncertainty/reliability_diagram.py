import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ------------------------
# Helper Functions
# ------------------------

def compute_bin_stats(confidences, correctness, num_bins=10):
    """
    Given flattened arrays of confidence values and binary correctness values, 
    compute the average confidence, average accuracy, and sample counts for each bin.

    Args:
        confidences (np.array): 1D array of predicted probabilities (range [0, 1]).
        correctness (np.array): 1D array of binary values (1 if voxel belongs to the class, else 0).
        num_bins (int): Number of bins to use.

    Returns:
        bin_centers (np.array): Center of each bin.
        avg_confidences (np.array): Average predicted probability in each bin.
        avg_accuracies (np.array): Average true accuracy in each bin.
        bin_counts (np.array): Number of samples in each bin.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2.0
    avg_confidences = np.zeros(num_bins)
    avg_accuracies = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (confidences >= lower) & (confidences < upper)
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            avg_confidences[i] = np.mean(confidences[in_bin])
            avg_accuracies[i] = np.mean(correctness[in_bin])
        else:
            avg_confidences[i] = np.nan
            avg_accuracies[i] = np.nan

    return bin_centers, avg_confidences, avg_accuracies, bin_counts

# ------------------------
# Main Script to Generate Reliability Diagrams per Sub-region
# ------------------------

def main():
    # Base directories for ground truth and predictions:
    gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
    pred_base_path = "../ensemble/output_segmentations/hybrid_new/"

    # List probability images. 
    # We assume filename pattern "ttd_softmax_{patientID}.nii.gz" where the probability map has shape (C, H, W, D)
    prob_images = sorted([f for f in os.listdir(pred_base_path) if re.match(r'hybrid_softmax_.*\.nii\.gz', f)])
    
    # Extract patient IDs from the probability filenames (assumes patient ID is 5 digits)
    pred_patient_ids = [re.search(r'hybrid_softmax_(\d{5})\.nii\.gz', f).group(1) for f in prob_images]
    
    # Filter ground truth images (assuming they are in folders or filenames that contain the patient IDs)
    # Here we assume that each ground truth is stored in a subfolder named after the patient (adjust as needed)
    gt_images = sorted([f for f in os.listdir(gt_base_path) if any(pid in f for pid in pred_patient_ids)])

    # We will aggregate per-voxel data for each class.
    # Here we assume class indices: 0 -> Background, 1 -> NCR, 2 -> ED, 3 -> ET.
    class_labels = {0: "Background", 1: "NCR", 2: "ED", 3: "ET"}
    num_classes = len(class_labels)

    # Initialize dictionaries to collect confidences and binary correctness.
    all_confidences = {c: [] for c in range(num_classes)}
    all_correctness = {c: [] for c in range(num_classes)}

    # Loop over patients. We align probability maps and ground truth based on ordering and patient IDs.
    for gt_image_name, prob_image_name in zip(gt_images, prob_images):
        # Construct ground truth path.
        # Here we assume that the GT file is in a subfolder named after the patient,
        # with the filename format "{patient}_seg.nii.gz" (adjust if needed).
        gt_image_folder = rf"{gt_image_name}\{gt_image_name}_seg.nii.gz"
        gt_image_path = os.path.join(gt_base_path, gt_image_folder)

        prob_image_path = os.path.join(pred_base_path, prob_image_name)

        try:
            gt_nii = nib.load(gt_image_path)
        except Exception as e:
            print(f"Error loading ground truth {gt_image_path}: {e}")
            continue
        gt_data = gt_nii.get_fdata()  # Expected shape: (H, W, D), integer labels.

        try:
            prob_nii = nib.load(prob_image_path)
        except Exception as e:
            print(f"Error loading probability image {prob_image_path}: {e}")
            continue
        softmax_prob = prob_nii.get_fdata()  # Expected shape: (C, H, W, D)
        if softmax_prob.ndim != 4:
            print(f"Unexpected shape for softmax image {prob_image_name}: {softmax_prob.shape}")
            continue

        # Loop over each class.
        for c in range(num_classes):
            # For class c, predicted probability is the softmax output for that channel.
            prob_c = softmax_prob[c, ...]
            # For calibration of a single class, the "correctness" is a binary indicator: 1 if gt equals the class, else 0.
            binary_label = (gt_data == c).astype(np.float32)

            # Flatten arrays and accumulate.
            all_confidences[c].append(prob_c.flatten())
            all_correctness[c].append(binary_label.flatten())

    # Concatenate arrays across all patients for each class.
    for c in range(num_classes):
        if all_confidences[c]:
            all_confidences[c] = np.concatenate(all_confidences[c])
            all_correctness[c] = np.concatenate(all_correctness[c])
        else:
            print(f"No data found for class {c} ({class_labels[c]}).")
            continue

    # Create reliability diagrams in a 2x2 subplot grid.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    num_bins = 10

    for c in range(num_classes):
        bin_centers, avg_conf, avg_acc, bin_counts = compute_bin_stats(all_confidences[c], all_correctness[c], num_bins)
        ax = axes[c]
        ax.plot(bin_centers, avg_acc, marker='o', linestyle='-', linewidth=2, label='Observed Accuracy')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.set_title(f"Reliability Diagram: {class_labels[c]}")
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left')
        ax.grid(True)
        # Annotate each bin with its count.
        for i in range(num_bins):
            if not np.isnan(avg_acc[i]):
                ax.text(bin_centers[i], avg_acc[i], f'{int(bin_counts[i])}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig("reliability_diagrams_subregions_hybrid.png")
    plt.show()

if __name__ == "__main__":
    main()
