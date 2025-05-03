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
    # gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\braintumor_data\VIGO_03"
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

def get_tumor_bounding_box(seg_data, slice_idx, padding=20):
    """
    Find the bounding box of the tumor region in a given slice.
    
    Args:
        seg_data: 3D segmentation array
        slice_idx: Slice index to analyze
        padding: Number of pixels to pad around the tumor
        
    Returns:
        xmin, xmax, ymin, ymax: Bounding box coordinates
    """
    # Get the slice and find non-zero (tumor) regions
    slice_data = seg_data[:, :, slice_idx]
    y_indices, x_indices = np.where(slice_data > 0)
    
    if len(y_indices) == 0:
        # If no tumor found, return the full image dimensions
        return 0, slice_data.shape[1], 0, slice_data.shape[0]
    
    # Calculate bounding box with padding
    xmin = max(0, x_indices.min() - padding)
    xmax = min(slice_data.shape[1], x_indices.max() + padding)
    ymin = max(0, y_indices.min() - padding)
    ymax = min(slice_data.shape[0], y_indices.max() + padding)
    
    return xmin, xmax, ymin, ymax

def plot_et_maps(data_paths, model_names, patient_id, slice_idx=None):
    """
    Plot ET probability maps, uncertainty maps, predicted and ground truth segmentations for multiple models.

    Args:
        data_paths: List of paths to the model outputs
        model_names: List of model names
        patient_id: Patient ID
        slice_idx: Optional slice index to plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Load ground truth data
    gt_base_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
    flair_path = os.path.join(gt_base_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_flair.nii.gz")
    flair_data = load_nifti(flair_path)
    seg_data = load_nifti(os.path.join(gt_base_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz"))

    if slice_idx is None:
        slice_idx = find_slice_max_label(patient_id, primary_label=2)[0]

    # Get tumor bounding box
    xmin, xmax, ymin, ymax = get_tumor_bounding_box(seg_data, slice_idx)

    # Create figure
    fig = plt.figure(figsize=(12, 3 * len(model_names)))
    gs = plt.GridSpec(len(model_names) + 1, 6, width_ratios=[0.3, 1, 1, 1, 1, 0.1], hspace=0.15, wspace=0.05)

    # Create custom colormap
    custom_cmap = ListedColormap(['black', 'red', 'yellow', 'blue'])

    # Add headers
    headers = ["Model", "ED Prob.", "ED Uncertainty", "Prediction", "Ground Truth"]
    for i, title in enumerate(headers):
        ax = fig.add_subplot(gs[0, i])
        ax.text(0.5, 0.3, title, ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.axis('off')

    for row, (model_name, data_path) in enumerate(zip(model_names, data_paths), start=1):
        flair_slice = flair_data[:, :, slice_idx]

        # Load model-specific data
        seg_path = f"{data_path}/{model_name}_{patient_id}_pred_seg.nii.gz"
        softmax_path = f"{data_path}/{model_name}_{patient_id}_softmax.nii.gz"
        unc_path = f"{data_path}/uncertainty_ED_{patient_id}.nii.gz"

        pred_seg = load_nifti(seg_path)[:, :, slice_idx]
        softmax = load_nifti(softmax_path)[2, :, :, slice_idx]  
        unc = load_nifti(unc_path)[:, :, slice_idx]
        gt_seg = seg_data[:, :, slice_idx]

        # Model name
        ax = fig.add_subplot(gs[row, 0])
        display_name = 'Hybrid' if model_name == 'hybrid_new' else model_name.upper()
        ax.text(0.5, 0.5, display_name, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.axis('off')

        # ET Prob
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(softmax, cmap='hot', alpha=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # ET Uncertainty
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(flair_slice, cmap='gray')
        im = ax.imshow(unc, cmap='hot', alpha=0.5, vmin=0.09, vmax=1)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # Prediction
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(pred_seg, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # Ground Truth
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(gt_seg, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.axis('off')

    # Colorbar
    cbar_ax = fig.add_subplot(gs[1:, 5])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Uncertainty/Probability', fontsize=11, fontweight='bold')

    # Save
    plt.suptitle(f"Patient {patient_id}", fontsize=14, fontweight='bold', y=0.85)
    plt.savefig(f"../visualization/uncertainty_maps/prob_vs_unc_ed_maps_{patient_id}_slice_{slice_idx}.png",
                bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    plt.close()

if __name__ == "__main__":
    patient_ids = ["00332", "01502"]
    models = ["tta", "ttd", "hybrid_new"]
    paths = ["../ensemble/output_segmentations/tta", 
             "../ensemble/output_segmentations/ttd", 
             "../ensemble/output_segmentations/hybrid_new"]
    
    for patient_id in patient_ids:
        if patient_id == "00332":
            plot_et_maps(paths, models, patient_id, slice_idx=115)
        
        if patient_id == "01502":
            plot_et_maps(paths, models, patient_id, slice_idx=104)
