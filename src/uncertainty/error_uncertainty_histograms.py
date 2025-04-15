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

# List predicted segmentation files (assumed filename pattern: {model}_{patient}_pred_seg.nii.gz)
pred_seg_files = sorted([f for f in os.listdir(pred_base_path) if re.match(rf'{model}_\d{{5}}_pred_seg\.nii\.gz', f)])
# List probability maps (assumed filename pattern: {model}_softmax_{patient}.nii.gz)
prob_files = sorted([f for f in os.listdir(pred_base_path) if re.match(rf'{model}_softmax_.*\.nii\.gz', f)])

# List uncertainty files for each sub-region (assumed filenames follow a pattern)
uncertainty_NCR_files = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_NCR.*_fused\.nii\.gz', f)])
uncertainty_ED_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ED.*_fused\.nii\.gz', f)])
uncertainty_ET_files  = sorted([f for f in os.listdir(pred_base_path) if re.match(r'uncertainty_ET.*_fused\.nii\.gz', f)])

# List GT folders (assuming each patient has a folder within gt_base_path)
gt_folders = sorted([f for f in os.listdir(gt_base_path) if os.path.isdir(os.path.join(gt_base_path, f))])

# Extract patient IDs from the predicted segmentation file names
patient_ids = [re.search(rf'{model}_(\d{{5}})_pred_seg', f).group(1) for f in pred_seg_files]

# Define sub-regions and their corresponding label values in the segmentation.
# For example, 1 = NCR, 2 = ED, 3 = ET (adjust if needed)
subregions = {
    'NCR': 1,
    'ED': 2,
    'ET': 3
}

# Define plotting style for each region
region_styles = {
    'NCR': {'color': 'dodgerblue', 'marker': 'o'},
    'ED': {'color': 'forestgreen', 'marker': 's'},
    'ET': {'color': 'crimson', 'marker': 'D'}
}

def plot_all_uncertainty_errors(bin_centers_dict, avg_error_dict, model_name):
    """Plot a single figure with the binned average error curves for all regions."""
    plt.figure(figsize=(10, 6))
    for region in bin_centers_dict.keys():
        bin_centers = bin_centers_dict[region]
        avg_error = avg_error_dict[region]
        style = region_styles.get(region, {'color': 'black', 'marker': 'o'})
        plt.plot(bin_centers, avg_error, marker=style['marker'], linestyle='-',
                 linewidth=3, markersize=8, color=style['color'], label=f'{region} Avg Error')
    
    plt.xlabel("Uncertainty (binned)", fontsize=16, fontweight='bold')
    plt.ylabel("Average Negative Log-Likelihood (Error)", fontsize=16, fontweight='bold')
    plt.title("Uncertainty vs Error for All Regions", fontsize=18, pad=20, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{model_name}_uncertainty_all_regions_corr_with_error_pretty.png", dpi=300)
    plt.close()

# This dictionary will accumulate voxel-level data for each sub-region.
# For each region we store:
#    - "error": the error computed per voxel from the probability map (negative log-likelihood)
#    - "uncertainty": the corresponding uncertainty value (from the fused uncertainty map)
vox_data = { region: {'error': [], 'uncertainty': []} for region in subregions.keys() }

eps = 1e-8  # To avoid log(0)

# Loop over patients. We assume consistent ordering for predicted segmentation, probability maps, and uncertainty maps.
for pid, seg_file, prob_file, unc_ncr_file, unc_ed_file, unc_et_file in zip(
        patient_ids, pred_seg_files, prob_files, uncertainty_NCR_files, uncertainty_ED_files, uncertainty_ET_files):

    # Find corresponding ground truth folder (assuming folder name contains the patient ID)
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

    # Load ground truth segmentation (shape: (H, W, D), integer labels)
    gt_data = nib.load(gt_file).get_fdata()

    # Load probability map (assumed shape: (C, H, W, D); C is the number of classes)
    prob_file_path = os.path.join(pred_base_path, prob_file)
    softmax_prob = nib.load(prob_file_path).get_fdata()

    # Load uncertainty maps for each sub-region
    unc_maps = {
        'NCR': nib.load(os.path.join(pred_base_path, unc_ncr_file)).get_fdata(),
        'ED':  nib.load(os.path.join(pred_base_path, unc_ed_file)).get_fdata(),
        'ET':  nib.load(os.path.join(pred_base_path, unc_et_file)).get_fdata()
    }

    # Loop through each sub-region
    for region, label in subregions.items():
        # Create a binary mask for voxels in the sub-region based on ground truth
        region_mask = (gt_data == label)
        if np.sum(region_mask) == 0:
            continue  # Skip this patient if region is absent

        # For computing error: extract the predicted probability for the true class from the softmax probability map.
        # Assume that the channel index corresponds directly to the label.
        pred_prob = softmax_prob[label, ...]  # Shape: (H, W, D)
        region_prob = pred_prob[region_mask]

        # Compute voxel-wise error as negative log-likelihood for the true class.
        region_error = -np.log(region_prob + eps)

        # Extract the corresponding uncertainty values from the uncertainty map for this region.
        region_uncertainty = unc_maps[region][region_mask]

        # Store the voxel values
        vox_data[region]['error'].append(region_error.flatten())
        vox_data[region]['uncertainty'].append(region_uncertainty.flatten())

# Dictionaries to hold binned data for each region for combined plotting.
all_bin_centers = {}
all_avg_errors = {}

# For each sub-region, aggregate voxel data across patients and visualize
for region in subregions.keys():
    if len(vox_data[region]['error']) == 0:
        print(f"No voxels found for region {region}")
        continue

    error_all = np.concatenate(vox_data[region]['error'])
    uncertainty_all = np.concatenate(vox_data[region]['uncertainty'])

    # Compute Spearman correlation between voxel-wise error and uncertainty
    corr, pval = spearmanr(error_all, uncertainty_all)
    print(f"\nRegion {region}:")
    print(f"Spearman correlation between error (NLL) and uncertainty: {corr:.4f}")
    print(f"P-value: {pval:.4f}")

    # Binned (aggregated) plot: Bin uncertainty values and plot average error per bin
    num_bins = 20
    bins = np.linspace(np.min(uncertainty_all), np.max(uncertainty_all), num_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    avg_error = []
    for i in range(num_bins):
        bin_mask = (uncertainty_all >= bins[i]) & (uncertainty_all < bins[i+1])
        if np.sum(bin_mask) > 0:
            avg_error.append(np.mean(error_all[bin_mask]))
        else:
            avg_error.append(np.nan)
    
    # Save binned data for later combined plotting
    all_bin_centers[region] = bin_centers
    all_avg_errors[region] = avg_error

    # Optionally, also visualize individual histograms (can be kept or removed)
    # plt.figure(figsize=(8, 4))
    # plt.hist(uncertainty_all, bins=50, alpha=0.7, label="Uncertainty")
    # plt.xlabel("Uncertainty")
    # plt.ylabel("Voxel Count")
    # plt.title(f"Histogram of Uncertainty for {region}")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8, 4))
    # plt.hist(error_all, bins=50, alpha=0.7, label="Error (NLL)", color='orange')
    # plt.xlabel("Error (NLL)")
    # plt.ylabel("Voxel Count")
    # plt.title(f"Histogram of Error for {region}")
    # plt.legend()
    # plt.show()

# Plot all sub-region uncertainty vs error curves on a single plot
plot_all_uncertainty_errors(all_bin_centers, all_avg_errors, model)
