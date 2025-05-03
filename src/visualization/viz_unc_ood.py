import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch 
import sys
import cv2
sys.path.append("..")
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from pathlib import Path

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


def load_nifti(filepath):
    """Load a NIfTI file and return its data array."""
    nifti_img = nib.load(filepath)
    return nifti_img.get_fdata()


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
                 "{model}_{patient}_pred_seg.nii.gz"

    Returns
    -------
    best_z     : int                # slice index
    diff_map   : 2-D np.ndarray     # boolean mask (True where models disagree)
    """
    # ---------- 1. load & stack predictions ----------------------------------
    vols = []
    for m in models:
        p = Path(base_paths[m]) / f"{m}_{patient_id}_pred_seg.nii.gz"
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

class ModelVisualizer:
    def __init__(self, model_type="ensemble"):
        """
        Initialize the ModelVisualizer.
        
        Args:
            model_type (str): Either "ensemble" or "single" to specify the type of model being visualized
        """
        self.model_type = model_type
        self.model_names = {
            "simple_avg": "SimpleAvg",
            "perf_weight": "PerfWeight",
            "tta": "TTA",
            "ttd": "TTD",
            "hybrid": "Hybrid",
            "gt": "Ground Truth",
            "flair": "FLAIR",
            "segresnet": "SegResNet",
            "attunet": "AttUNet",
            "swinunetr": "SwinUNETR"
        }
        
        if model_type == "ensemble":
            self.base_paths = {
                "simple_avg": "../stats/segmentations/simple_avg",
                "perf_weight": "../stats/segmentations/pwe",
                "tta": "../stats/segmentations/tta",
                "ttd": "../stats/segmentations/ttd",
                "hybrid": "../stats/segmentations/hybrid"
            }
        else:
            self.base_paths = {
                "segresnet": "../models/predictions/segresnet",
                "attunet": "../models/predictions/attunet",
                "swinunetr": "../models/predictions/swinunetr"
            }

    def get_model_path(self, model_name):
        """Get the path for a specific model's predictions."""
        return self.base_paths.get(model_name)

    def load_prediction(self, model_name, patient_id):
        """Load prediction for a specific model and patient."""
        path = self.get_model_path(model_name)
        if not path:
            raise ValueError(f"Unknown model: {model_name}")
            
        pred_path = os.path.join(path, f"{model_name}_{patient_id}_pred_seg.nii.gz")
        return load_nifti(pred_path)

    def load_ground_truth(self, patient_id):
        """Load ground truth for a patient."""
        gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"
        gt_seg_path = os.path.join(gt_base_path, f"{patient_id}_seg.nii.gz")
        return load_nifti(gt_seg_path)

    def load_flair(self, patient_id):
        """Load FLAIR image for a patient."""
        gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"
        flair_path = os.path.join(gt_base_path, "original", f"preprocessed1/preproc_{patient_id}_FLAIR_orig_skullstripped.nii.gz")
        return load_nifti(flair_path)

    def find_best_slice(self, patient_id, model_name=None, primary_label=1, fallback_labels=[2, 3]):
        """Find the best slice for visualization."""
        return find_slice_max_label(patient_id, model_name, self.get_model_path(model_name), primary_label, fallback_labels)

    def load_uncertainty(self, model_name: str, patient_id: str, region: str):
        """
        Load the (already saved) voxel-wise uncertainty map for a given model,
        patient and tumour sub-region (ED, ET, NCR).

        Expected file name:  uncertainty_<REGION>_<PID>.nii.gz
        """
        path = self.get_model_path(model_name)
        fname = f"uncertainty_{region}_{patient_id}.nii.gz"
        return load_nifti(os.path.join(path, fname))
    
    def create_uncertainty_figure(self,
                              patients: list[str],
                              models:   list[str],
                              regions:  list[str] = ("NCR", "ED", "ET"),
                              save_path: str | None = None):
        """
        For every patient create one figure:
            rows   = models (e.g. TTD, TTA, Hybrid)
            cols   = tumour sub-regions (NCR, ED, ET)
            cells  = uncertainty overlay on FLAIR
        """
        target_size = (256, 256)
        unc_cmap    = get_cmap("hot")
        norm        = Normalize(vmin=0.0, vmax=1.0)     # used for colour-bar

        for pid in patients:
            # -------- choose slice & bounding box --------------------------------
            best_z, _ = most_disagreeing_slice(pid, models, self.base_paths)
            flair_vol = self.load_flair(pid)
            gt_vol    = self.load_ground_truth(pid)
            gt_slice  = gt_vol[:, :, best_z]
            bbox      = get_tumor_bbox(gt_slice, padding=40) \
                        or (0, gt_slice.shape[0], 0, gt_slice.shape[1])
            y0, y1, x0, x1 = bbox

            # -------- figure & gridspec -----------------------------------------
            n_rows = len(models)          # models on the vertical axis
            n_cols = len(regions) + 1     # +1 narrow label column
            fig   = plt.figure(figsize=(3.3 * n_cols, 3.0 * n_rows))
            gs    = plt.GridSpec(n_rows, n_cols,
                                figure=fig,
                                height_ratios=[1]*n_rows,
                                width_ratios=[0.25] + [1]*len(regions))

            fig.suptitle(f"Uncertainty maps – {pid}", fontsize=22, y=1.02)

            # -------- iterate rows(models) / cols(regions) ----------------------
            for r, model in enumerate(models):
                # left-hand label cell (row titles = model names)
                ax_lbl = fig.add_subplot(gs[r, 0])
                ax_lbl.axis("off")
                ax_lbl.text(0.5, 0.5, self.model_names[model],
                            ha="center", va="center",
                            fontsize=14, fontweight="bold", rotation=90)

                # load once per model to save I/O
                unc_vols = {reg: self.load_uncertainty(model, pid, reg) for reg in regions}

                for c, region in enumerate(regions, start=1):
                    ax = fig.add_subplot(gs[r, c])

                    unc_slice = unc_vols[region][:, :, best_z][y0:y1, x0:x1]
                    flair_c   = flair_vol[:, :, best_z][y0:y1, x0:x1]

                    flair_rs  = resize_to_fixed_size(flair_c, target_size)
                    unc_rs    = resize_to_fixed_size(unc_slice, target_size)

                    # 0-1 normalisation per slice for visual contrast
                    if unc_rs.max() > unc_rs.min():
                        unc_norm = (unc_rs - unc_rs.min()) / (unc_rs.max() - unc_rs.min())
                    else:
                        unc_norm = unc_rs

                    ax.imshow(flair_rs, cmap="gray")
                    ax.imshow(unc_norm, cmap=unc_cmap, alpha=0.5, vmin=0.09, vmax=1)
                    ax.axis("off")

                    if r == 0:          # column headers (region names)
                        ax.set_title(region, fontsize=16, fontweight="bold")

            # -------- unified colour-bar ----------------------------------------
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])     # [left, bottom, w, h]
            sm  = plt.cm.ScalarMappable(cmap=unc_cmap, norm=norm)
            sm.set_array([])
            cb  = fig.colorbar(sm, cax=cax)
            cb.set_label("Uncertainty", rotation=270, labelpad=15, fontsize=12)

            plt.subplots_adjust(wspace=0.05, hspace=0.05)

            # -------- save / show ----------------------------------------------
            if save_path:
                base, ext = os.path.splitext(save_path)
                ext = ext if ext else ".png"
                out = f"{base}_{pid}{ext}" if ext else f"{save_path}_{pid}.png"
                plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.show()
            plt.close(fig)

    def create_comparison_figure(self, patients, models, save_path=None):
        """
        Create a comparison figure for multiple patients and models.
        
        Args:
            patients (list): List of patient IDs
            models (list): List of model names to compare
            save_path (str, optional): Path to save the figure
        """
        # Create figure with 2 rows and len(models) + 2 columns (1 for patient names, len(models) for models, 1 for GT, 1 for FLAIR)
        fig = plt.figure(figsize=(32, 10))
        gs = plt.GridSpec(2, len(models) + 3, figure=fig, height_ratios=[1, 1], 
                         width_ratios=[0.1] + [1] * (len(models) + 2))
        
        # Create custom colormap for segmentation regions
        from matplotlib.colors import ListedColormap
        colors = ['black', 'red', 'yellow', 'blue']  # Background, NCR, ED, ET
        custom_cmap = ListedColormap(colors)
        
        # Fixed size for all images
        target_size = (256, 256)
        
        # Process each patient
        for row, patient_id in enumerate(patients):
            # Add patient name in first column
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.text(0.5, 0.5, patient_id, 
                         ha="center", va="center", 
                         fontsize=20, fontweight='bold',
                         rotation=90,
                         transform=ax_label.transAxes)
            ax_label.axis("off")
            
            # Get data for all models
            all_data = []
            for model in models:
                pred_seg_data = self.load_prediction(model, patient_id)
                seg_data = self.load_ground_truth(patient_id)
                flair_data = self.load_flair(patient_id)
                # slice_idx = self.find_best_slice(patient_id, model)[0]
                best_z, diff_map = most_disagreeing_slice(patient_id, models, self.base_paths)
                slice_idx = best_z
                all_data.append((pred_seg_data, seg_data, flair_data, slice_idx))
            
            # Get ground truth slice and its bounding box for consistent cropping
            gt_slice = all_data[0][1][:, :, all_data[0][3]]
            bbox = get_tumor_bbox(gt_slice, padding=40)
            
            if bbox is None:
                print(f"No tumor found in ground truth for patient {patient_id}")
                continue
                
            y_min, y_max, x_min, x_max = bbox
            
            # Plot predictions for each model
            for col, (model, (pred_seg_data, _, flair_data, slice_idx)) in enumerate(zip(models, all_data)):
                ax = fig.add_subplot(gs[row, col + 1])  # +1 to skip patient name column
                
                # Extract and crop slices
                flair_slice = flair_data[:, :, slice_idx]
                pred_seg_slice = pred_seg_data[:, :, slice_idx]
                
                flair_crop = flair_slice[y_min:y_max, x_min:x_max]
                pred_seg_crop = pred_seg_slice[y_min:y_max, x_min:x_max]
                
                # Resize cropped images to target size
                flair_resized = resize_to_fixed_size(flair_crop, target_size)
                pred_seg_resized = resize_segmentation(pred_seg_crop, target_size)
                
                ax.imshow(flair_resized, cmap='gray')
                ax.imshow(pred_seg_resized, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
                
                # Add title with metrics if requested
                if row == 0:  # Only add titles for the first row
                    title = self.model_names[model]
                    ax.set_title(title, fontsize=20, fontweight='bold', pad=10)
                ax.axis('off')
                ax.set_aspect('equal')
            
            # Plot ground truth
            ax = fig.add_subplot(gs[row, len(models) + 1])  # Ground truth is second to last column
            flair_slice = all_data[0][2][:, :, all_data[0][3]]
            seg_slice = all_data[0][1][:, :, all_data[0][3]]
            
            # Crop and resize
            flair_crop = flair_slice[y_min:y_max, x_min:x_max]
            seg_crop = seg_slice[y_min:y_max, x_min:x_max]
            
            flair_resized = resize_to_fixed_size(flair_crop, target_size)
            seg_resized = resize_segmentation(seg_crop, target_size)
            
            ax.imshow(flair_resized, cmap='gray')
            ax.imshow(seg_resized, cmap=custom_cmap, alpha=0.5, vmin=0, vmax=3)
            if row == 0:  # Only add titles for the first row
                ax.set_title(self.model_names["gt"], fontsize=20, fontweight='bold', pad=10)
            ax.axis('off')
            ax.set_aspect('equal')
            
            # Plot raw FLAIR
            ax = fig.add_subplot(gs[row, len(models) + 2])  # FLAIR is last column
            flair_slice = all_data[0][2][:, :, all_data[0][3]]
            flair_crop = flair_slice[y_min:y_max, x_min:x_max]
            flair_resized = resize_to_fixed_size(flair_crop, target_size)
            
            ax.imshow(flair_resized, cmap='gray')
            if row == 0:  # Only add titles for the first row
                ax.set_title(self.model_names["flair"], fontsize=20, fontweight='bold', pad=10)
            ax.axis('off')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Example usage for single models
    viz = ModelVisualizer(model_type="ensemble")
    single_models = ["ttd", "tta", "hybrid"]

    patients = ["VIGO_01", "VIGO_03"]
    
    # Create visualization with metrics
    viz.create_uncertainty_figure(
        patients=patients,
        models=single_models,
        save_path="../visualization/ood_preds/uncertainty_maps.png",
    )
