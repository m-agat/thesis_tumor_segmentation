import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.style.use('seaborn-v0_8-whitegrid')

# Base paths for ground truth and predicted outputs
gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"

model = "tta"
pred_base_path = f"../ensemble/output_segmentations/{model}/"

# List predicted segmentation files (assumed filename pattern: ttd_{patient}_pred_seg.nii.gz)
pred_seg_files = sorted([f for f in os.listdir(pred_base_path) if re.match(rf'{model}_\d{{5}}_pred_seg\.nii\.gz', f)])
# List probability maps (assumed filename pattern: ttd_softmax_{patient}.nii.gz)
prob_files = sorted([f for f in os.listdir(pred_base_path) if re.match(rf'{model}_softmax_.*\.nii\.gz', f)])
# List uncertainty files for each sub-region (assumed filenames follow a pattern)
uncertainty_NCR_files = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_NCR.*_fused\.nii\.gz', f)])
uncertainty_ED_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ED.*_fused\.nii\.gz', f)])
uncertainty_ET_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ET.*_fused\.nii\.gz', f)])
# List GT folders (assuming each patient has a folder within gt_base_path)
gt_folders = sorted([f for f in os.listdir(gt_base_path) if os.path.isdir(os.path.join(gt_base_path, f))])
# Extract patient IDs from the predicted segmentation file names
patient_ids = [re.search(rf'{model}_(\d{{5}})_pred_seg', f).group(1) for f in pred_seg_files]

# Define sub-regions and their corresponding label values in the segmentation:
# Adjust labels as needed (e.g., 1 = NCR, 2 = ED, 3 = ET)
subregions = {
    'NCR': 1,
    'ED': 2,
    'ET': 3
}

region_styles = {
    'NCR': {'color': 'dodgerblue', 'marker': 'o'},
    'ED': {'color': 'forestgreen', 'marker': 's'},
    'ET': {'color': 'crimson', 'marker': 'D'}
}

# This dictionary will accumulate voxel-level data for each sub-region.
# For each region, we store:
#   - "error": computed using negative log-likelihood error (NLL) from probability maps.
#   - "uncertainty": corresponding uncertainty values from the uncertainty map.
vox_data = { region: {'error': [], 'uncertainty': []} for region in subregions.keys() }

eps = 1e-8   # To avoid log(0)

# Loop over patients; assume the ordering in all lists is consistent.
for pid, seg_file, prob_file, unc_ncr_file, unc_ed_file, unc_et_file in zip(
        patient_ids, pred_seg_files, prob_files, uncertainty_NCR_files, uncertainty_ED_files, uncertainty_ET_files):

    # Find the corresponding ground truth folder by patient ID.
    gt_folder_candidates = [f for f in gt_folders if pid in f]
    if not gt_folder_candidates:
        print(f"Ground truth folder not found for patient {pid}")
        continue
    gt_folder = gt_folder_candidates[0]
    # Construct GT segmentation file path (assumed pattern: {folder}/{folder}_seg.nii.gz)
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

    # For each sub-region, extract voxel-level error and uncertainty.
    for region, label in subregions.items():
        # Create a mask for voxels in the current sub-region using ground truth.
        region_mask = (gt_data == label)
        if np.sum(region_mask) == 0:
            continue

        # Extract the predicted probability for the given class.
        pred_prob = softmax_prob[label, ...]   # Assumes the channel index corresponds to the label.
        # Compute negative log-likelihood (NLL) error.
        region_error = -np.log(pred_prob[region_mask] + eps)
        # Extract uncertainty values for the region.
        region_uncertainty = unc_maps[region][region_mask]

        # Append flattened arrays.
        vox_data[region]['error'].append(region_error.flatten())
        vox_data[region]['uncertainty'].append(region_uncertainty.flatten())

# Function to compute risk–coverage curve.
def compute_risk_coverage(uncertainty, error):
    """
    Given arrays of uncertainty and error (both 1D),
    compute risk (average error) as a function of coverage.
    Coverage is defined as the fraction of voxels with the lowest uncertainty.
    """
    # Sort voxels by uncertainty (ascending).
    sorted_indices = np.argsort(uncertainty)
    uncertainty_sorted = uncertainty[sorted_indices]
    error_sorted = error[sorted_indices]

    n_vox = len(uncertainty_sorted)
    # Define coverage fractions from 0 to 1.
    coverage_fractions = np.linspace(0.05, 1.0, 20)  # e.g., 5% to 100% in 20 steps
    risk = []  # Average error for the selected voxels.

    for frac in coverage_fractions:
        n_keep = int(n_vox * frac)
        if n_keep == 0:
            risk.append(np.nan)
        else:
            # Compute average error in the voxels with the lowest uncertainty.
            risk.append(np.mean(error_sorted[:n_keep]))
    return coverage_fractions, risk

fig, ax = plt.subplots(figsize=(10, 6))

# For each sub-region, aggregate voxel data and compute risk–coverage curves.
for region in subregions.keys():
    if len(vox_data[region]['error']) == 0:
        print(f"No voxels found for region {region}")
        continue

    error_all = np.concatenate(vox_data[region]['error'])
    uncertainty_all = np.concatenate(vox_data[region]['uncertainty'])
    
    # Compute the risk-coverage curve
    coverage, risk = compute_risk_coverage(uncertainty_all, error_all)
    
    # Plotting with customized styles
    ax.plot(
        coverage,
        risk,
        marker=region_styles[region]['marker'],
        linestyle='-',
        color=region_styles[region]['color'],
        label=f"{region}"
    )

    # Also, print out the Spearman correlation (for reference).
    corr, pval = spearmanr(error_all, uncertainty_all)
    print(f"\nRegion {region}: Spearman correlation between error and uncertainty: {corr:.4f} (p={pval:.4f})")

# Set plot labels and title
ax.set_xlabel("Coverage Fraction (Voxels with Lowest Uncertainty)", fontsize=14, fontweight='bold')
ax.set_ylabel("Average Error (Negative Log-Likelihood)", fontsize=14, fontweight='bold')
ax.set_title("Risk–Coverage Curves by Tumor Sub-region", fontsize=16, fontweight='bold')

# Add legend and adjust its properties
ax.legend(title="Regions", fontsize=12, title_fontsize=13, loc='upper left')

# Adjust tick parameters for readability
ax.tick_params(axis='both', which='major', labelsize=12)

# Add grid for visual clarity
ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)

# Tight layout for better use of space
plt.tight_layout()

plt.savefig(f"./Figures/{model}_risk_coverage.png", dpi=300)

# Display plot
plt.show()

plt.close()

    
