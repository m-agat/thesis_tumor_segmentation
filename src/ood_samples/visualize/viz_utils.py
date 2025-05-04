import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
from pathlib import Path
import sys

def load_nifti(filepath):
    """Load a NIfTI file and return its data array."""
    nifti_img = nib.load(filepath)
    return nifti_img.get_fdata()

def resize_to_fixed_size(image, target_size=(256, 256)):
    """Resize image to a fixed size using cv2."""
    return cv2.resize(image.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)

def resize_segmentation(seg_image, target_size=(256, 256)):
    """Resize segmentation mask while preserving labels."""
    resized = np.zeros((target_size[0], target_size[1]), dtype=seg_image.dtype)
    for label in np.unique(seg_image):
        if label == 0:  # Skip background
            continue
        # Create binary mask for this label
        mask = (seg_image == label).astype(np.float32)
        # Resize the binary mask
        resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)
        # Threshold to get back binary mask
        resized_mask = (resized_mask > 0.5).astype(seg_image.dtype)
        # Add to final image
        resized[resized_mask == 1] = label
    return resized

def find_slice_max_label(patient_id, model_name, path, primary_label=1, fallback_labels=[2, 3]):
    """
    For a given patient, find the slice index in the ground truth segmentation volume that has the
    maximum number of voxels for the primary label (default: NCR=1). If no voxel with the primary 
    label is found in any slice, then fallback to the slices with the highest count for fallback labels.
    
    Returns:
      best_slice: Selected slice index.
      used_label: The label that produced the maximum count.
    """
    # Load ground truth segmentation instead of prediction
    gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"
    gt_seg_path = os.path.join(gt_base_path, f"{patient_id}_seg.nii.gz")
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


def most_disagreeing_slice(patient_id: str,
                           models: list[str],
                           base_paths: dict[str, str]) -> tuple[int, np.ndarray]:
    """
    Return the z–index with the greatest inter-model disagreement and the
    per-voxel disagreement map for that slice.

    Parameters
    ----------
    patient_id : str
    models     : list of model keys (e.g. ["simple_avg", "tta", …])
    base_paths : mapping model_key -> directory that contains files named
                 "seg_{patient}.nii.gz"

    Returns
    -------
    best_z     : int                # slice index
    diff_map   : 2-D np.ndarray     # boolean mask (True where models disagree)
    """
    # ---------- 1. load & stack predictions ----------------------------------
    vols = []
    for m in models:
        p = Path(base_paths[m]) / f"seg_{patient_id}.nii.gz"
        vols.append(nib.load(p).get_fdata().astype(np.int16))   # (H,W,D)
    stack = np.stack(vols, axis=0)                              # (M,H,W,D)

    # ---------- 2. compute disagreement score for every slice ----------------
    # max-min > 0  -> at least two different labels at that voxel
    disagree = (stack.max(0) - stack.min(0)) > 0                # (H,W,D)
    scores   = disagree.sum(axis=(0,1))                         # (D,)
    best_z   = int(scores.argmax())

    return best_z, disagree[..., best_z]


def crop_to_tumor_area(image, seg, target_label=1, padding=30):
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

def compute_used_label_for_slice(gt_slice):
    """
    Given a ground truth segmentation slice, compute which tumor label (from 1, 2, or 3)
    is the most dominant. Returns the label with the maximum count (if any), or None if none exists.
    """
    counts = {label: np.sum(gt_slice == label) for label in [1, 2, 3]}
    # Choose label with maximum count.
    used_label = max(counts, key=counts.get)
    return used_label if counts[used_label] > 0 else None

def get_tumor_bbox(seg_slice, padding=20):
    """Get bounding box around tumor region with padding."""
    # Find coordinates of non-zero pixels (tumor)
    y_coords, x_coords = np.nonzero(seg_slice > 0)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return None
        
    # Get bounding box with padding
    y_min = max(0, np.min(y_coords) - padding)
    y_max = min(seg_slice.shape[0], np.max(y_coords) + padding)
    x_min = max(0, np.min(x_coords) - padding)
    x_max = min(seg_slice.shape[1], np.max(x_coords) + padding)
    
    return (y_min, y_max, x_min, x_max)