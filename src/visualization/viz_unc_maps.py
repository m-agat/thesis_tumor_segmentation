import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 

def load_nifti(filepath):
    """Load a NIfTI file and return its data array."""
    nifti_img = nib.load(filepath)
    return nifti_img.get_fdata()

def find_slice_max_label(patient_id, primary_label=1, fallback_labels=[2, 3]):
    """
    For a given patient, find the slice index in the GT segmentation volume that has the
    maximum number of voxels for the primary label (default: NCR=1). If no voxel with the primary 
    label is found in any slice, then fallback to the slices with the highest count for fallback labels.
    
    Returns:
      best_slice: Selected slice index.
      used_label: The label that produced the maximum count.
    """
    gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
    gt_seg_path = os.path.join(gt_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz")
    gt_seg = nib.load(gt_seg_path).get_fdata().astype(np.int32)
    best_slice = None
    best_count = 0
    used_label = primary_label

    # First, try with primary label.
    for slice_idx in range(gt_seg.shape[2]):
        count = np.sum(gt_seg[:, :, slice_idx] == primary_label)
        if count > best_count:
            best_count = count
            best_slice = slice_idx

    # If no voxels for primary label exist, try fallback labels in order.
    if best_count == 0:
        for label in fallback_labels:
            for slice_idx in range(gt_seg.shape[2]):
                count = np.sum(gt_seg[:, :, slice_idx] == label)
                if count > best_count:
                    best_count = count
                    best_slice = slice_idx
                    used_label = label
            if best_count > 0:
                break

    # If still no tumor found, default to middle slice.
    if best_slice is None:
        best_slice = gt_seg.shape[2] // 2
        used_label = None

    return best_slice, used_label

def plot_uncertainty_maps(data_path, model_name, patient_id, slice_idx=None):
    seg_path = f"{data_path}/{model_name}_{patient_id}_pred_seg.nii.gz"
    gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
    flair_path = os.path.join(gt_base_path, f"BraTS2021_{patient_id}/BraTS2021_{patient_id}_flair.nii.gz")
    pred_seg_data = load_nifti(seg_path)
    flair_data = load_nifti(flair_path)
    seg_data = load_nifti(os.path.join(gt_base_path, f"BraTS2021_{patient_id}/BraTS2021_{patient_id}_seg.nii.gz"))

    if slice_idx is None:
        slice_idx = find_slice_max_label(patient_id)[0]
    
    # Load uncertainty data for each region
    region_names = ['NCR', 'ED', 'ET']
    unc_data_dict = {}
    for region in region_names:
        unc_path = f"{data_path}/uncertainty_{region}_{patient_id}_fused.nii.gz"
        unc_data_dict[region] = load_nifti(unc_path)

    # Determine global min and max across all regions (for consistent color scale)
    all_unc_values = np.concatenate([
        unc_data_dict[r].flatten() for r in region_names
    ])
    global_min = np.nanmin(all_unc_values)
    global_max = np.nanmax(all_unc_values)

    # Create figure with custom layout to accommodate colorbar
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1])  # Last column for colorbar
    
    # Create axes for the main plots
    axes = []
    for i in range(2):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))
    
    # Plot uncertainty maps in first row
    for i, region in enumerate(region_names):
        flair_slice = flair_data[:, :, slice_idx]
        unc_slice = unc_data_dict[region][:, :, slice_idx]

        axes[i].imshow(flair_slice, cmap='gray')
        # Overlay the uncertainty using a semi-transparent overlay
        im = axes[i].imshow(
            unc_slice,
            cmap='hot',
            alpha=0.5,
            vmin=0.09,
            vmax=1
        )
        axes[i].set_title(f"{region} Uncertainty", fontsize=13, fontweight='bold')
        axes[i].axis('off')

    # Add colorbar spanning both rows
    cbar_ax = fig.add_subplot(gs[:, 3])  # Span all rows in the last column
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Uncertainty', fontsize=13, fontweight='bold')

    # Create custom colormap for segmentation regions
    from matplotlib.colors import ListedColormap
    colors = ['black', 'red', 'yellow', 'blue']  # Background, NCR, ED, ET
    custom_cmap = ListedColormap(colors)

    # Plot predicted segmentation
    pred_seg_slice = pred_seg_data[:, :, slice_idx]
    axes[3].imshow(flair_slice, cmap='gray')
    axes[3].imshow(pred_seg_slice, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
    axes[3].set_title("Predicted Segmentation", fontsize=13, fontweight='bold')
    axes[3].axis('off')

    # Plot ground truth segmentation
    seg_slice = seg_data[:, :, slice_idx]
    axes[4].imshow(flair_slice, cmap='gray')
    axes[4].imshow(seg_slice, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
    axes[4].set_title("Ground Truth Segmentation", fontsize=13, fontweight='bold')
    axes[4].axis('off')

    # Plot raw FLAIR image
    axes[5].imshow(flair_slice, cmap='gray')
    axes[5].set_title("FLAIR Image", fontsize=13, fontweight='bold')
    axes[5].axis('off')
    # Add padding between suptitle and plots
    plt.suptitle(f"Patient {patient_id}", fontsize=15, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the top margin to make space for suptitle
    plt.savefig(f"../visualization/uncertainty_maps/uncertainty_map_{model_name}_{patient_id}_slice_{slice_idx}.png")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    patient_ids = ["00332", "01502", "01147", "01483"]
    models = ["tta", "ttd"]
    paths = ["../ensemble/output_segmentations/tta", "../ensemble/output_segmentations/ttd"]
    
    for patient_id in patient_ids:
        for model, path in zip(models, paths):
            plot_uncertainty_maps(path, model, patient_id)

