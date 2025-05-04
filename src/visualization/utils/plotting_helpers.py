import numpy as np
from pathlib import Path
import os
import nibabel as nib
from visualization.utils.io import load_nifti
from visualization.utils.constants import *

def find_slice_max_label(patient_id, primary_label=1, fallback_labels=[2, 3]):
    """
    For a given patient, find the slice index in the GT segmentation volume that has the
    maximum number of voxels for the primary label (default: NCR=1). If no voxel with the primary 
    label is found in any slice, then fallback to the slices with the highest count for fallback labels.
    
    Returns:
      best_slice: Selected slice index.
      used_label: The label that produced the maximum count.
    """
    gt_seg_path = os.path.join(DATA_ROOT, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz")
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

def crop_to_bbox(img: np.ndarray, bbox: tuple[int,int,int,int]):
    y0,y1,x0,x1 = bbox
    return img[y0:y1, x0:x1]

def add_significance_markers(ax, means, sems, significant_pairs, model_names):
    if not significant_pairs:
        return

    ymin, ymax = ax.get_ylim()
    span        = ymax - ymin
    base_offset = 0.1 * span    
    label_offset = 0.01 * span     

    # used_positions tracks the top of each bar + its label
    used_positions = {i: [] for i in range(len(model_names))}
    for i, (mean, sem) in enumerate(zip(means, sems)):
        used_positions[i].append(mean + sem + label_offset)

    pairs = sorted(
        significant_pairs,
        key=lambda p: abs(model_names.index(p[0]) - model_names.index(p[1])),
        reverse=True
    )

    for g1, g2 in pairs:
        i1, i2 = model_names.index(g1), model_names.index(g2)
        lo, hi = min(i1, i2), max(i1, i2)

        # look at the _actual_ registered heights
        block_max = max(max(used_positions[x]) for x in range(lo, hi+1))

        # nudge up until no collision
        level = block_max + base_offset
        conflict = True
        while conflict:
            conflict = False
            for x in range(lo, hi+1):
                for used in used_positions[x]:
                    if abs(level - used) < (base_offset * 0.8):
                        level += base_offset
                        conflict = True
                        break
                if conflict:
                    break

        # draw bars and star
        ax.plot([i1, i2], [level, level], 'k-', lw=1, clip_on=False)
        ax.plot([i1, i1], [level - label_offset, level], 'k-', lw=1, clip_on=False)
        ax.plot([i2, i2], [level - label_offset, level], 'k-', lw=1, clip_on=False)
        ax.text((i1+i2)/2, level + (0.5 * label_offset), '*',
                ha='center', va='bottom', fontsize=12, clip_on=False)

        # register that new level
        for x in range(lo, hi+1):
            used_positions[x].append(level)

    # extend y-limit so none of that is clipped
    new_max = max(max(v) for v in used_positions.values())
    ax.set_ylim(ymin, new_max + base_offset)
