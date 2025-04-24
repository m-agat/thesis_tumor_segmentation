import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os 
import torch 
import sys
import cv2
sys.path.append("..")
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import config.config as config 

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

def compute_metrics(pred, gt):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    This version is specifically for visualization purposes.
    Assumes pred and gt are each a single tensor of shape [C, H, W, D].
    """
    assert isinstance(pred, torch.Tensor) and pred.ndim == 5 and pred.shape[0] == 1, f"Expected [1, C, H, W, D], got {pred.shape}"
    assert isinstance(gt, torch.Tensor) and gt.ndim == 5 and gt.shape[0] == 1, f"Expected [1, C, H, W, D], got {gt.shape}"

    pred = pred.squeeze(0).detach().clone().to(config.device)  # -> [C, H, W, D]
    gt = gt.squeeze(0).detach().clone().to(config.device)

    # MONAI metrics expect list of batch items
    pred_list = [pred.squeeze(0)]
    gt_list = [gt.squeeze(0)]

    # Dice Score
    dice_metric = DiceMetric(
        include_background=False, reduction=MetricReduction.NONE, get_not_nans=True
    )
    dice_metric(y_pred=pred_list, y=gt_list)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.cpu().numpy()
    not_nans = not_nans.cpu().numpy()

    pred_channels = list(torch.unbind(pred.squeeze(0), dim=0))  # [C, H, W, D]
    gt_channels = list(torch.unbind(gt.squeeze(0), dim=0))

    for i in range(len(dice_scores[0])):
        if not_nans[0][i] == 0:  # Tissue absent in GT
            pred_empty = torch.sum(pred_channels[i]) == 0
            dice_scores[0][i] = 1.0 if pred_empty else 0.0

    # HD95
    hd95 = compute_hausdorff_distance(
        y_pred=pred.unsqueeze(0),
        y=gt.unsqueeze(0),
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
    )
    hd95 = hd95.squeeze(0).cpu().numpy()

    for i in range(len(hd95)):
        pred_empty = torch.sum(pred_channels[i]) == 0
        gt_empty = not_nans[0][i] == 0

        if pred_empty and gt_empty:
            print(f"Region {i}: Both GT and Prediction are empty. Setting HD95 to 0.")
            hd95[i] = 0.0
        elif gt_empty and not pred_empty:
            pred_np = pred_channels[i].cpu().numpy()
            if np.sum(pred_np) > 0:
                com = center_of_mass(pred_np)
                com_mask = np.zeros_like(pred_np, dtype=np.uint8)
                com_mask[tuple(map(int, map(round, com)))] = 1
                com_tensor = torch.from_numpy(com_mask).float().to(config.device).unsqueeze(0).unsqueeze(0)
                mock_val = compute_hausdorff_distance(
                    y_pred=pred[i].unsqueeze(0).unsqueeze(0),  # [1, 1, H, W, D]
                    y=com_tensor,                              # [1, 1, H, W, D]
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95
                )
                hd95[i] = mock_val.squeeze().item()
        elif pred_empty and not gt_empty:
            gt_np = gt_channels[i].cpu().numpy()
            if np.sum(gt_np) > 0:
                com = center_of_mass(gt_np)
                com_mask = np.zeros_like(gt_np, dtype=np.uint8)
                com_mask[tuple(map(int, map(round, com)))] = 1
                com_tensor = torch.from_numpy(com_mask).float().to(config.device).unsqueeze(0).unsqueeze(0)
                mock_val = compute_hausdorff_distance(
                    y_pred=pred[i].unsqueeze(0).unsqueeze(0),  # [1, 1, H, W, D]
                    y=com_tensor,                              # [1, 1, H, W, D]
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95
                )
                hd95[i] = mock_val.squeeze().item()

    # Sensitivity & Specificity
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity"],
        reduction="none",
        compute_sample=False,
    )
    confusion_metric(y_pred=pred_list, y=gt_list)
    sensitivity, specificity = confusion_metric.aggregate()
    sensitivity = sensitivity.squeeze(0).cpu().numpy()
    specificity = specificity.squeeze(0).cpu().numpy()

    for i in range(len(sensitivity)):
        if not_nans[0][i] == 0:
            pred_empty = torch.sum(pred_channels[i]) == 0
            sensitivity[i] = 1.0 if pred_empty else 0.0
            specificity[i] = 1.0

    return dice_scores, hd95, sensitivity, specificity

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

def plot_uncertainty_maps(data_path, model_name, patient_id, slice_idx=None):
    seg_path = f"{data_path}/{model_name}_{patient_id}_pred_seg.nii.gz"
    # gt_base_path = rf"\\wsl.localhost\Ubuntu-22.04\home\magata\data\braintumor_data\{patient_id}"
    gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"

    flair_path = os.path.join(gt_base_path, "original" f"/preprocessed1/preproc_{patient_id}_FLAIR_orig_skullstripped.nii.gz")
    pred_seg_data = load_nifti(seg_path)
    flair_data = load_nifti(flair_path)
    gt_seg_path = os.path.join(gt_base_path, f"{patient_id}_seg.nii.gz")
    seg_data = load_nifti(gt_seg_path)

    # Convert predictions and ground truth to one-hot format
    pred_one_hot = [(pred_seg_data == i).astype(float) for i in range(4)]
    gt_one_hot = [(seg_data == i).astype(float) for i in range(4)]
    
    # Stack into tensors
    pred_one_hot = torch.from_numpy(np.stack(pred_one_hot))
    gt_one_hot = torch.from_numpy(np.stack(gt_one_hot))

    pred_one_hot = pred_one_hot.unsqueeze(0)
    gt_one_hot = gt_one_hot.unsqueeze(0)

    # Compute metrics
    dice, hd95, sensitivity, specificity = compute_metrics(pred_one_hot, gt_one_hot)
    print("model: ", model_name)
    print("Dice: ", dice)
    print("HD95: ", hd95)
    print("Sensitivity: ", sensitivity)
    print("Specificity: ", specificity, "\n")

    if slice_idx is None:
        slice_idx = find_slice_max_label(patient_id, model_name, data_path)[0]

    return pred_seg_data, seg_data, flair_data, slice_idx

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

    def compute_metrics(self, pred, gt):
        """Compute metrics for a prediction and ground truth pair."""
        return compute_metrics(pred, gt)

    def find_best_slice(self, patient_id, model_name=None, primary_label=1, fallback_labels=[2, 3]):
        """Find the best slice for visualization."""
        return find_slice_max_label(patient_id, model_name, self.get_model_path(model_name), primary_label, fallback_labels)

    def compute_model_metrics(self, model_name, patient_id):
        """
        Compute metrics for a specific model and patient.
        
        Returns:
            dict: Dictionary containing Dice, HD95, Sensitivity, and Specificity scores
        """
        pred_data = self.load_prediction(model_name, patient_id)
        gt_data = self.load_ground_truth(patient_id)
        
        # Convert to one-hot format
        pred_one_hot = [(pred_data == i).astype(float) for i in range(4)]
        gt_one_hot = [(gt_data == i).astype(float) for i in range(4)]
        
        # Stack into tensors
        pred_one_hot = torch.from_numpy(np.stack(pred_one_hot))
        gt_one_hot = torch.from_numpy(np.stack(gt_one_hot))
        
        pred_one_hot = pred_one_hot.unsqueeze(0)
        gt_one_hot = gt_one_hot.unsqueeze(0)
        
        # Compute metrics
        dice, hd95, sensitivity, specificity = self.compute_metrics(pred_one_hot, gt_one_hot)
        
        return {
            "dice": dice.squeeze(0).cpu().numpy(),
            "hd95": hd95.squeeze(0).cpu().numpy(),
            "sensitivity": sensitivity.squeeze(0).cpu().numpy(),
            "specificity": specificity.squeeze(0).cpu().numpy()
        }

    def compute_all_metrics(self, models, patients):
        """
        Compute metrics for all models and patients.
        
        Returns:
            dict: Dictionary containing metrics for each model and patient
        """
        all_metrics = {}
        for model in models:
            model_metrics = {}
            for patient in patients:
                metrics = self.compute_model_metrics(model, patient)
                model_metrics[patient] = metrics
            all_metrics[model] = model_metrics
        return all_metrics

    def print_metrics(self, metrics):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics (dict): Dictionary containing metrics for each model and patient
        """
        for model, model_metrics in metrics.items():
            print(f"\nModel: {self.model_names[model]}")
            print("-" * 50)
            
            for patient, patient_metrics in model_metrics.items():
                print(f"\nPatient: {patient}")
                print(f"Dice Scores:")
                print(f"  NCR: {patient_metrics['dice'][1]:.4f}")
                print(f"  ED: {patient_metrics['dice'][2]:.4f}")
                print(f"  ET: {patient_metrics['dice'][3]:.4f}")
                
                print(f"\nHD95 Scores:")
                print(f"  NCR: {patient_metrics['hd95'][1]:.2f}")
                print(f"  ED: {patient_metrics['hd95'][2]:.2f}")
                print(f"  ET: {patient_metrics['hd95'][3]:.2f}")
                
                print(f"\nSensitivity Scores:")
                print(f"  NCR: {patient_metrics['sensitivity'][1]:.4f}")
                print(f"  ED: {patient_metrics['sensitivity'][2]:.4f}")
                print(f"  ET: {patient_metrics['sensitivity'][3]:.4f}")
                
                print(f"\nSpecificity Scores:")
                print(f"  NCR: {patient_metrics['specificity'][1]:.4f}")
                print(f"  ED: {patient_metrics['specificity'][2]:.4f}")
                print(f"  ET: {patient_metrics['specificity'][3]:.4f}")
                print("-" * 30)

    def create_comparison_figure(self, patients, models, save_path=None, show_metrics=False):
        """
        Create a comparison figure for multiple patients and models.
        
        Args:
            patients (list): List of patient IDs
            models (list): List of model names to compare
            save_path (str, optional): Path to save the figure
            show_metrics (bool): Whether to show metrics in the figure
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
        
        # Compute metrics if requested
        if show_metrics:
            metrics = self.compute_all_metrics(models, patients)
        
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
                slice_idx = self.find_best_slice(patient_id, model)[0]
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
                    if show_metrics:
                        metrics_text = f"\nDice: {metrics[model][patient_id]['dice'][1:].mean():.3f}"
                        title += metrics_text
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
    single_visualizer = ModelVisualizer(model_type="single")
    single_models = ["segresnet", "attunet", "swinunetr"]
    patients = ["VIGO_01", "VIGO_03"]
    
    # Compute and print metrics
    metrics = single_visualizer.compute_all_metrics(single_models, patients)
    single_visualizer.print_metrics(metrics)
    
    # Create visualization with metrics
    single_visualizer.create_comparison_figure(
        patients=patients,
        models=single_models,
        save_path="../visualization/ood_preds/comparison_figure_single.png",
        show_metrics=True
    )
