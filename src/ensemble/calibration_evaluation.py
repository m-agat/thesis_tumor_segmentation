import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os 
import json
import re

############################
# Basic Utility Functions
############################

def extract_patient_id(filepath):
    """
    Extract patient ID from a file path using a regex.
    For example, from "RelabeledTrainingData/BraTS2021_00403/BraTS2021_00403_flair.nii.gz",
    it extracts "00403".
    """
    match = re.search(r'BraTS2021_(\d+)', filepath)
    if match:
        return match.group(1)
    return None

def compute_error_map(pred_binary, gt_binary):
    """
    Given binary masks (1=region, 0=not-region),
    compute a binary error map (1 for error, 0 for correct).
    """
    return (pred_binary != gt_binary).astype(np.float32)

def compute_reliability_data(uncertainty, error_map, n_bins=10):
    """
    Compute bin centers, mean predicted uncertainty, and observed error rates.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    mean_conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)
    for i in range(n_bins):
        bin_mask = (uncertainty >= bins[i]) & (uncertainty < bins[i+1])
        if np.sum(bin_mask) > 0:
            mean_conf[i] = np.mean(uncertainty[bin_mask])
            acc[i] = np.mean(error_map[bin_mask])
        else:
            mean_conf[i] = bin_centers[i]
            acc[i] = np.nan
    return bin_centers, mean_conf, acc

def plot_reliability_diagram(bin_centers, mean_conf, acc, title="Reliability Diagram", save_path="reliability_diagram.png"):
    plt.figure(figsize=(6,6))
    plt.plot(bin_centers, acc, marker='o', linestyle='-', label="Observed Error")
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label="Perfect Calibration")
    plt.xlabel("Mean Predicted Uncertainty")
    plt.ylabel("Observed Error Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def compute_ece(mean_conf, acc, error_counts, n_total):
    """
    Compute Expected Calibration Error.
    error_counts: number of voxels in each bin
    """
    ece = 0.0
    for i in range(len(mean_conf)):
        if error_counts[i] > 0:
            ece += (error_counts[i] / n_total) * abs(acc[i] - mean_conf[i])
    return ece


############################
# Main Aggregation Function
############################

def aggregate_uncertainty_and_error_subregions(patient_list, subregion_map, uncertainty_dir, pred_dir, gt_dir):
    """
    For each patient, we:
      1) Load predicted labels.
      2) Load ground truth labels.
      3) For each subregion name (including background):
         - Load the corresponding uncertainty file: e.g. "new_uncertainty_BG_{patient}.nii.gz"
         - Create a mask from the ground truth (e.g., true_labels == label_value)
         - Create a predicted mask (pred_labels == label_value)
         - Compute binary error map for that subregion.
         - Gather the uncertainty values ONLY for voxels belonging to that region in the ground truth.
    We store all flattened uncertainties and error_map values per subregion in a dictionary.
    """

    # Prepare storage
    data_dict = {subregion_name: {"unc": [], "err": []}
                 for subregion_name in subregion_map.keys()}

    for patient in patient_list:
        # Load predicted segmentation and ground truth
        pred_path = os.path.join(pred_dir, f"hybrid_segmentation_{patient}.nii.gz")
        gt_path = os.path.join(gt_dir, f"BraTS2021_{patient}", f"BraTS2021_{patient}_seg.nii.gz")

        if not (os.path.exists(pred_path) and os.path.exists(gt_path)):
            print(f"Skipping {patient}, missing pred/gt files.")
            continue

        pred_labels = nib.load(pred_path).get_fdata()
        true_labels = nib.load(gt_path).get_fdata()

        # For each subregion (including BG), load the subregion-specific uncertainty
        for subregion_name, label_value in subregion_map.items():
            unc_path = os.path.join(uncertainty_dir, f"new_uncertainty_{subregion_name}_{patient}.nii.gz")
            if not os.path.exists(unc_path):
                print(f"Skipping {patient}, missing {unc_path}.")
                continue

            uncertainty_map = nib.load(unc_path).get_fdata()

            # Create a ground-truth mask for this subregion
            gt_mask = (true_labels == label_value)
            if not np.any(gt_mask):
                # No such subregion in this patient
                continue

            # Create a predicted mask for the same subregion
            pred_mask = (pred_labels == label_value)

            # Compute error map for this subregion
            error_map = compute_error_map(pred_mask, gt_mask)

            # Filter the uncertainty and error_map to the subregion in ground truth
            subregion_unc = uncertainty_map[gt_mask]
            subregion_err = error_map[gt_mask]

            # Flatten and store
            data_dict[subregion_name]["unc"].append(subregion_unc.flatten())
            data_dict[subregion_name]["err"].append(subregion_err.flatten())

    # Now concatenate for each subregion
    aggregated_data = {}
    for subregion_name in data_dict:
        if len(data_dict[subregion_name]["unc"]) == 0:
            aggregated_data[subregion_name] = (np.array([]), np.array([]))
        else:
            unc_cat = np.concatenate(data_dict[subregion_name]["unc"])
            err_cat = np.concatenate(data_dict[subregion_name]["err"])
            aggregated_data[subregion_name] = (unc_cat, err_cat)

    return aggregated_data


############################
# Example Usage
############################

def evaluate_calibration_dual(uncertainty, error_map, gt_mask, pred_mask, region, n_bins=10):
    """
    Compute and plot reliability diagrams and ECE for a given region using:
      - The ground truth mask (GT evaluation)
      - The predicted mask (Pred evaluation)
    
    Parameters:
      uncertainty: full array of uncertainty values (e.g., flattened or full volume)
      error_map: corresponding binary error array (1 for error, 0 for correct)
      gt_mask: boolean array (same shape as uncertainty) where True indicates voxels belonging to the region per ground truth
      pred_mask: boolean array (same shape as uncertainty) where True indicates voxels predicted as that region
      region: name of the region (e.g., "NCR")
      n_bins: number of bins for the reliability diagram
      
    Returns:
      (gt_ece, pred_ece): tuple of ECE values for the GT-based evaluation and the predicted-based evaluation
    """
    # Evaluate on ground truth mask
    gt_unc = uncertainty[gt_mask]
    gt_err = error_map[gt_mask]
    if gt_unc.size == 0:
        print(f"No voxels in ground truth mask for region {region}.")
        gt_ece = None
    else:
        gt_bin_centers, gt_mean_conf, gt_acc = compute_reliability_data(gt_unc, gt_err, n_bins)
        bins = np.linspace(0, 1, n_bins + 1)
        gt_error_counts = np.array([np.sum((gt_unc >= bins[i]) & (gt_unc < bins[i+1])) for i in range(n_bins)])
        gt_n_total = gt_unc.size
        gt_ece = compute_ece(gt_mean_conf, gt_acc, gt_error_counts, gt_n_total)
        plot_reliability_diagram(
            gt_bin_centers,
            gt_mean_conf,
            gt_acc,
            title=f"Reliability Diagram ({region} - GT)",
            save_path=f"reliability_diagram_{region}_GT.png"
        )

    # Evaluate on predicted mask
    pred_unc = uncertainty[pred_mask]
    pred_err = error_map[pred_mask]
    if pred_unc.size == 0:
        print(f"No voxels in predicted mask for region {region}.")
        pred_ece = None
    else:
        pred_bin_centers, pred_mean_conf, pred_acc = compute_reliability_data(pred_unc, pred_err, n_bins)
        bins = np.linspace(0, 1, n_bins + 1)
        pred_error_counts = np.array([np.sum((pred_unc >= bins[i]) & (pred_unc < bins[i+1])) for i in range(n_bins)])
        pred_n_total = pred_unc.size
        pred_ece = compute_ece(pred_mean_conf, pred_acc, pred_error_counts, pred_n_total)
        plot_reliability_diagram(
            pred_bin_centers,
            pred_mean_conf,
            pred_acc,
            title=f"Reliability Diagram ({region} - Pred)",
            save_path=f"reliability_diagram_{region}_Pred.png"
        )

    print(f"ECE for {region} (GT): {gt_ece:.3f}" if gt_ece is not None else f"ECE for {region} (GT): None")
    print(f"ECE for {region} (Pred): {pred_ece:.3f}" if pred_ece is not None else f"ECE for {region} (Pred): None")
    return gt_ece, pred_ece


if __name__ == "__main__":
    # Directories (adjust as needed)
    uncertainty_dir = "./output_segmentations/hybrid/"
    pred_dir = "./output_segmentations/hybrid/"
    gt_dir = r"/home/magata/data/brats2021challenge/RelabeledTrainingData/"

    # Subregion map: name -> label_value
    # Here we include BG (background) as label 0.
    # In BraTS: 0 = BG, 1 = NCR, 2 = ED, 3 = ET
    subregion_map = {
        "BG": 0,
        "NCR": 1,
        "ED": 2,
        "ET": 3
    }

    patient_list = ["00483"]

    # Aggregate data for each subregion, now including BG.
    aggregated_data = aggregate_uncertainty_and_error_subregions(
        patient_list, subregion_map, uncertainty_dir, pred_dir, gt_dir
    )

    # For each subregion, compute reliability diagram and ECE.
    for subregion_name, (unc_array, err_array) in aggregated_data.items():
        if len(unc_array) == 0:
            print(f"No data found for {subregion_name}, skipping.")
            continue

        print(f"\n=== Subregion: {subregion_name} ===")
        print(f"Uncertainty shape: {unc_array.shape}, Error shape: {err_array.shape}")

        # Compute reliability
        n_bins = 10
        bin_centers, mean_conf, acc = compute_reliability_data(unc_array, err_array, n_bins=n_bins)

        # Compute ECE
        bins = np.linspace(0, 1, n_bins + 1)
        error_counts = np.array([
            np.sum((unc_array >= bins[i]) & (unc_array < bins[i+1]))
            for i in range(n_bins)
        ])
        n_total = unc_array.size
        ece = compute_ece(mean_conf, acc, error_counts, n_total)
        print(f"Expected Calibration Error (ECE) for {subregion_name}: {ece:.3f}")

        # Plot reliability diagram
        plot_reliability_diagram(
            bin_centers,
            mean_conf,
            acc,
            title=f"Reliability Diagram ({subregion_name})",
            save_path=f"reliability_diagram_{subregion_name}_new.png"
        )
