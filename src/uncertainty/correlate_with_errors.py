import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Base paths for ground truth and predicted outputs
gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
pred_base_path = "../ensemble/output_segmentations/tta/"

# List predicted segmentation files (assumed filename pattern: tta_{patient}_pred_seg.nii.gz)
pred_seg_files = sorted([f for f in os.listdir(pred_base_path) if re.match(r'tta_\d{5}_pred_seg\.nii\.gz', f)])

# List probability maps (assumed filename pattern: tta_softmax_{patient}.nii.gz)
prob_files = sorted([f for f in os.listdir(pred_base_path) if re.match(r'tta_softmax_.*\.nii\.gz', f)])

# List uncertainty files for each sub-region (assumed filenames follow a pattern)
uncertainty_NCR_files = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_NCR.*_fused\.nii\.gz', f)])
uncertainty_ED_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ED.*_fused\.nii\.gz', f)])
uncertainty_ET_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ET.*_fused\.nii\.gz', f)])

# List GT folders (assuming each patient has a folder within gt_base_path)
gt_folders = sorted([f for f in os.listdir(gt_base_path) if os.path.isdir(os.path.join(gt_base_path, f))])

# Extract patient IDs from the predicted segmentation file names
patient_ids = [re.search(r'tta_(\d{5})_pred_seg', f).group(1) for f in pred_seg_files]

# Define sub-regions and their corresponding label values in the segmentation:
# Adjust the labels as needed (e.g., 1 = NCR, 2 = ED, 3 = ET)
subregions = {
    'NCR': 1,
    'ED': 2,
    'ET': 3
}

# This dictionary accumulates voxel-level data for each sub-region.
# For each region, we store lists for:
#   - "error": computed from probability maps (negative log-likelihood)
#   - "uncertainty": corresponding uncertainty values from the uncertainty map.
vox_data = { region: {'error': [], 'uncertainty': []} for region in subregions.keys() }

eps = 1e-8  # Small epsilon to avoid log(0)

# Loop over patients; we assume the ordering in all lists is consistent.
for pid, seg_file, prob_file, unc_ncr_file, unc_ed_file, unc_et_file in zip(
        patient_ids, pred_seg_files, prob_files, uncertainty_NCR_files, uncertainty_ED_files, uncertainty_ET_files):

    # Find the corresponding ground truth folder (check for patient id in folder name).
    gt_folder_candidates = [f for f in gt_folders if pid in f]
    if not gt_folder_candidates:
        print(f"Ground truth folder not found for patient {pid}")
        continue
    gt_folder = gt_folder_candidates[0]
    # Construct the GT segmentation file path (assumed pattern: {folder}/{folder}_seg.nii.gz)
    gt_file = os.path.join(gt_base_path, gt_folder, f"{gt_folder}_seg.nii.gz")
    if not os.path.exists(gt_file):
        print(f"Ground truth file not found: {gt_file}")
        continue

    # Load ground truth segmentation (expected shape: (H, W, D), integer labels)
    gt_data = nib.load(gt_file).get_fdata()

    # Load probability map (expected shape: (C, H, W, D))
    prob_file_path = os.path.join(pred_base_path, prob_file)
    softmax_prob = nib.load(prob_file_path).get_fdata()

    # Load uncertainty maps for each sub-region.
    unc_maps = {
        'NCR': nib.load(os.path.join(pred_base_path, unc_ncr_file)).get_fdata(),
        'ED':  nib.load(os.path.join(pred_base_path, unc_ed_file)).get_fdata(),
        'ET':  nib.load(os.path.join(pred_base_path, unc_et_file)).get_fdata()
    }

    # For each sub-region, compute voxel-level error and collect uncertainty.
    for region, label in subregions.items():
        # Create mask for voxels in the current sub-region using ground truth.
        region_mask = (gt_data == label)
        if np.sum(region_mask) == 0:
            continue  # Skip if region not present

        # From the probability map, extract the predicted probability for class "label".
        # It is assumed that the channel index corresponds to the label.
        pred_prob = softmax_prob[label, ...]  # Shape: (H, W, D)

        # Compute error for voxels in the region using negative log-likelihood.
        region_error = -np.log(pred_prob[region_mask] + eps)

        # Extract corresponding uncertainty values from the uncertainty map.
        region_uncertainty = unc_maps[region][region_mask]

        # Append flattened arrays to cumulative lists.
        vox_data[region]['error'].append(region_error.flatten())
        vox_data[region]['uncertainty'].append(region_uncertainty.flatten())

# Now, for each sub-region, aggregate voxel-level data across patients,
# and visualize the relationship using both scatter plots and percentile-based binned plots.
for region in subregions.keys():
    if len(vox_data[region]['error']) == 0:
        print(f"No voxels found for region {region}")
        continue

    error_all = np.concatenate(vox_data[region]['error'])
    uncertainty_all = np.concatenate(vox_data[region]['uncertainty'])

    # Compute Spearman correlation between error and uncertainty.
    corr, pval = spearmanr(error_all, uncertainty_all)
    print(f"\nRegion {region}:")
    print(f"Spearman correlation between error (NLL) and uncertainty: {corr:.4f}")
    print(f"P-value: {pval:.4f}")

    # Scatter plot with log-scale for error to help visualize the skew.
    plt.figure(figsize=(8, 4))
    plt.scatter(uncertainty_all, error_all, s=1, alpha=0.05)
    plt.xlabel("Uncertainty")
    plt.ylabel("Error (NLL)")
    plt.title(f"Voxel-wise Analysis for {region} (log-scaled error)")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    # Binned (aggregated) plot using percentile-based bins.
    num_bins = 20
    percentiles = np.linspace(0, 100, num_bins+1)
    bin_edges = np.percentile(uncertainty_all, percentiles)
    bin_centers = []
    median_error = []
    for i in range(num_bins):
        lower, upper = bin_edges[i], bin_edges[i+1]
        bin_mask = (uncertainty_all >= lower) & (uncertainty_all < upper)
        if np.sum(bin_mask) > 0:
            bin_centers.append((lower+upper)/2)
            median_error.append(np.median(error_all[bin_mask]))
        else:
            bin_centers.append((lower+upper)/2)
            median_error.append(np.nan)
    plt.figure(figsize=(8, 4))
    plt.plot(bin_centers, median_error, marker='o', linestyle='-')
    plt.xlabel("Uncertainty (percentile bin center)")
    plt.ylabel("Median Error (NLL)")
    plt.title(f"Binned Analysis (Percentile-based) for {region}")
    plt.grid(True)
    plt.show()

    # Optionally, visualize histograms (using log-scale for error histogram)
    plt.figure(figsize=(8, 4))
    plt.hist(uncertainty_all, bins=50, alpha=0.7, label="Uncertainty")
    plt.xlabel("Uncertainty")
    plt.ylabel("Voxel Count")
    plt.title(f"Histogram of Uncertainty for {region}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(error_all, bins=50, alpha=0.7, label="Error (NLL)", color='orange')
    plt.xlabel("Error (NLL)")
    plt.ylabel("Voxel Count")
    plt.title(f"Histogram of Error for {region} (log scale)", fontsize=10)
    plt.xscale('log')
    plt.legend()
    plt.show()
