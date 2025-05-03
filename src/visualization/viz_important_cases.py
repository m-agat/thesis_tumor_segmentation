import os 
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# Cases definition
unc_better    = "00332"  
simple_better = "01032"  
good_case     = "01147"  
worst_case    = "01483"

attunet_case    = "01556"  # best at segmenting NCR
segresnet_case  = "01474"  # good at segmenting ET and ED
swinunetr_case = "01405"
borderline_case = "01529"  # most challenging

# Optionally, you might want to override the slice index per case.
# If provided here, these take precedence.
slice_indices = {
    unc_better: 120,
    simple_better: 115,
    good_case: 65,
    worst_case: 80
}

# For labeling the columns, you can add descriptive titles.
cases = [
    (unc_better, ""),
    (simple_better, ""),
    (good_case, ""),
    (worst_case, "")
]

cases_indiv = [
    (attunet_case, ""),
    (segresnet_case, ""),
    (swinunetr_case, ""),
    (borderline_case, "")
]

gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
# pred_base = "../ensemble/output_segmentations"
pred_base = "../models/predictions"

models_dict_names = {
    "simple_avg": "SimpleAvg",
    "perf_weight": "PerfWeight",
    "ttd": "TTD",
    "tta": "TTA",
    "hybrid_new": "Hybrid"
}

# Define the order and names for the rows.
sources = [
    ("simple_avg", models_dict_names["simple_avg"]),
    ("perf_weight", models_dict_names["perf_weight"]),
    ("ttd", models_dict_names["ttd"]),
    ("tta", models_dict_names["tta"]),
    ("hybrid_new", models_dict_names["hybrid_new"]),
    ("gt", "GT")
]

models_dict_names_indiv = {
    "vnet": "V-Net",
    "segresnet": "SegResNet",
    "attunet": "Attention UNet",
    "swinunetr": "SwinUNETR"
}

sources_indiv = [
    ("vnet", models_dict_names_indiv["vnet"]),
    ("segresnet", models_dict_names_indiv["segresnet"]),
    ("attunet", models_dict_names_indiv["attunet"]),
    ("swinunetr", models_dict_names_indiv["swinunetr"]),
    ("gt", "Ground Truth")
]

def create_overlay(seg_slice, overlay_colors):
    """
    Create a full overlay image (RGBA) from the 2D segmentation slice.
    
    Parameters:
      seg_slice: a 2D numpy array containing integer labels.
      overlay_colors: dictionary mapping label to an RGBA tuple.
    
    Returns:
      overlay: an RGBA image of the same height and width as seg_slice.
    """
    overlay = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 4), dtype=float)
    for label, color in overlay_colors.items():
        mask = seg_slice == label
        overlay[mask] = color
    return overlay

def load_t1ce_and_seg(patient_id, slice_index):
    """
    Loads the T1ce slice and ground truth segmentation slice for a given patient.
    Returns: t1ce_slice (2D numpy array), gt_seg_slice (2D numpy array)
    """
    t1ce_path = os.path.join(gt_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_t1ce.nii.gz")
    gt_seg_path = os.path.join(gt_path, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz")
    
    # Load T1ce image and extract slice.
    t1ce_img = nib.load(t1ce_path)
    t1ce_data = t1ce_img.get_fdata()
    t1ce_slice = t1ce_data[:, :, slice_index]
    
    # Load Ground Truth segmentation.
    gt_seg = nib.load(gt_seg_path).get_fdata().astype(np.int32)
    gt_seg_slice = gt_seg[:, :, slice_index]
    
    return t1ce_slice, gt_seg_slice

def load_prediction(patient_id, model, slice_index):
    """
    Loads the predicted segmentation for the given model and patient and extracts the required slice.
    """
    pred_seg_path = os.path.join(pred_base, model, f"{model}_{patient_id}_pred_seg.nii.gz")
    seg = nib.load(pred_seg_path).get_fdata().astype(np.int32)
    return seg[:, :, slice_index]

def find_slice_max_label(patient_id, primary_label=1, fallback_labels=[2, 3]):
    """
    For a given patient, find the slice index in the GT segmentation volume that has the
    maximum number of voxels for the primary label (default: NCR=1). If no voxel with the primary 
    label is found in any slice, then fallback to the slices with the highest count for fallback labels.
    
    Returns:
      best_slice: Selected slice index.
      used_label: The label that produced the maximum count.
    """
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

def compute_used_label_for_slice(gt_slice):
    """
    Given a ground truth segmentation slice, compute which tumor label (from 1, 2, or 3)
    is the most dominant. Returns the label with the maximum count (if any), or None if none exists.
    """
    counts = {label: np.sum(gt_slice == label) for label in [1, 2, 3]}
    # Choose label with maximum count.
    used_label = max(counts, key=counts.get)
    return used_label if counts[used_label] > 0 else None

def crop_to_tumor_area(image, seg, target_label=1, padding=10):
    """
    Crop the input image to the bounding box of the tumor area in the segmentation,
    using the specified target label.
    
    Returns the cropped image and bounding box coordinates (y0, y1, x0, x1).
    If no tumor is found for the target_label, returns the original image and full coordinates.
    """
    mask = seg == target_label
    if np.sum(mask) == 0:
        # If no tumor with the specified label is found, return the original image.
        return image, (0, image.shape[0], 0, image.shape[1])
    
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # add one to include the max index
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(image.shape[0], y1 + padding)
    x1 = min(image.shape[1], x1 + padding)
    return image[y0:y1, x0:x1], (y0, y1, x0, x1)

# Define font sizes for better visibility
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12

# Set global font sizes
plt.rcParams.update({
    'font.size': LABEL_FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_FONT_SIZE,
    'ytick.labelsize': TICK_FONT_SIZE,
    'legend.fontsize': LABEL_FONT_SIZE
})

# Make fonts bold for better visibility
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

def visualize_with_gridspec(cases, sources, slice_indices):
    overlay_colors = {
        1: (1, 0, 0, 0.8),   # NCR: red
        2: (1, 1, 0, 0.8),   # ED: yellow
        3: (0, 0, 1, 0.8)    # ET: blue
    }

    n_rows = len(sources)
    n_cols_cases = len(cases)
    total_cols = n_cols_cases + 1  # first column for model labels

    fig = plt.figure(figsize=(20, 15))
    width_ratios = [0.2] + [1.5] * n_cols_cases
    gs = GridSpec(
        nrows=n_rows, ncols=total_cols,
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.02,
        hspace=0.02
    )
    
    for row, (source_key, model_label) in enumerate(sources):
        ax_label = fig.add_subplot(gs[row, 0], rasterized=True)
        ax_label.text(
            0.0, 0.5, model_label,
            ha="left", va="center",
            fontsize=14,
            transform=ax_label.transAxes
        )
        ax_label.axis("off")
        
        for col, (patient_id, _) in enumerate(cases, start=1):
            # Select the slice:
            # If a slice index is provided, use it, but also determine the dominant tumor label in that slice.
            if patient_id in slice_indices:
                slice_index = slice_indices[patient_id]
                # Load GT slice to compute the dominant label.
                _, gt_seg_slice = load_t1ce_and_seg(patient_id, slice_index)
                used_label = compute_used_label_for_slice(gt_seg_slice)
                if used_label is None:
                    used_label = 1  # fallback to NCR if none is found
            else:
                slice_index, used_label = find_slice_max_label(patient_id, primary_label=1, fallback_labels=[2, 3])
            
            # Load T1ce and GT segmentation slices.
            t1ce_slice, gt_seg_slice = load_t1ce_and_seg(patient_id, slice_index)
            # Crop the images to zoom in on the tumor area based on the dominant tumor label.
            t1ce_crop, bbox = crop_to_tumor_area(t1ce_slice, gt_seg_slice, target_label=used_label, padding=30)
            gt_seg_crop = gt_seg_slice[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            
            # Load the segmentation for the given source.
            if source_key == "gt":
                seg_slice = gt_seg_slice
            else:
                seg_slice = load_prediction(patient_id, source_key, slice_index)
            # Crop the segmentation to the same bounding box.
            seg_crop = seg_slice[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            # Create overlay on the cropped segmentation.
            overlay = create_overlay(seg_crop, overlay_colors)
            
            ax = fig.add_subplot(gs[row, col])
            # Only add title for the first row (row == 0)
            if row == 0:
                if cases[col-1][1]:  # If there's a title specified
                    ax.set_title(cases[col-1][1], pad=10)
                else:  # Otherwise use patient ID
                    ax.set_title(f"Patient {patient_id}", pad=10)
            ax.imshow(t1ce_crop, cmap="gray", interpolation="none")
            ax.imshow(overlay, interpolation="none")
            ax.axis("off")

    fig.subplots_adjust(
        left=0.02, right=0.98, 
        top=0.96, bottom=0.04,
        wspace=0.02, hspace=0.02
    )
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("example_cases_indiv.png", dpi=300, bbox_inches="tight")
    plt.show()

# Create the composite figure.
visualize_with_gridspec(cases_indiv, sources_indiv, {})
