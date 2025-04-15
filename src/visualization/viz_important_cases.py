import os 
import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

# Cases definition
unc_better    = "00332"  
simple_better  = "01075"  
good_case = "01147"  
worst_case = "01483"

# Optionally, you might want to use different slice indices per case:
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

gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
# pred_base = "../models/predictions"
pred_base = "../ensemble/output_segmentations"

# Dictionary mapping model keys to display names.
# models_dict_names = {
#     "vnet": "V-Net",
#     "segresnet": "SegResNet",
#     "attunet": "Attention UNet",
#     "swinunetr": "SwinUNETR"
# }
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
    ("gt", "Ground Truth")
]

def create_overlay(seg_slice, overlay_colors):
    """
    Create a full overlay image (RGBA) from the 2D segmentation slice.
    
    Parameters:
    - seg_slice: a 2D numpy array containing integer labels.
    - overlay_colors: dictionary mapping label to an RGBA tuple.
    
    Returns:
    - overlay: an RGBA image of the same height and width as seg_slice.
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


def visualize_with_gridspec(cases, sources, slice_indices):
    overlay_colors = {
        1: (1, 0, 0, 0.8),   
        2: (1, 1, 0, 0.8),   
        3: (0, 0, 1, 0.8)    
    }

    n_rows = len(sources)
    n_cols_cases = len(cases)
    total_cols = n_cols_cases + 1

    # Increase the figure size significantly
    fig = plt.figure(figsize=(20, 15))
    
    # Make the label column slightly wider for better readability
    width_ratios = [0.08] + [1.5]*n_cols_cases
    
    # Create a GridSpec with minimal wspace
    gs = GridSpec(
        nrows=n_rows, ncols=total_cols,
        figure=fig, 
        width_ratios=width_ratios,
        wspace=0.02,   # slightly increased horizontal space
        hspace=0.02   # minimal vertical space
    )
    
    for row, (source_key, model_label) in enumerate(sources):
        # Left label axis
        ax_label = fig.add_subplot(gs[row, 0], rasterized=True)
        ax_label.text(
            0.0, 0.5, model_label,
            ha="left", va="center",
            fontsize=14,  # Increased font size
            transform=ax_label.transAxes
        )
        ax_label.axis("off")
        
        for col, (patient_id, _) in enumerate(cases, start=1):
            slice_index = slice_indices.get(patient_id, 80)
            t1ce_slice, gt_seg_slice = load_t1ce_and_seg(patient_id, slice_index)
            
            if source_key == "gt":
                seg_slice = gt_seg_slice
            else:
                seg_slice = load_prediction(patient_id, source_key, slice_index)

            overlay = create_overlay(seg_slice, overlay_colors)
            
            ax = fig.add_subplot(gs[row, col])
            # Reduced the amount we shrink the subplots to make them bigger
            pos = ax.get_position()
            delta_w, delta_h = pos.width * 0.02, pos.height * 0.02
            new_pos = [pos.x0 + delta_w, pos.y0 + delta_h, pos.width - 2*delta_w, pos.height - 2*delta_h]
            ax.set_position(new_pos)
            ax.imshow(t1ce_slice, cmap="gray", interpolation="none")
            ax.imshow(overlay, interpolation="none")
            ax.axis("off")

    # Adjust the overall figure margins to maximize space for subplots
    fig.subplots_adjust(
        left=0.02, right=0.98, 
        top=0.96, bottom=0.04,
        wspace=0.02, hspace=0.02
    )
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Slightly increased padding
    plt.savefig("three_cases_composite_gridspec.png", dpi=300, bbox_inches="tight")
    plt.show()

# Create the composite figure.
visualize_with_gridspec(cases, sources, slice_indices)