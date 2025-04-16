import os
import re
import numpy as np
import nibabel as nib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# Define model and paths
model = "tta"
gt_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
pred_path = f"../ensemble/output_segmentations/{model}"

# Find all predicted segmentation files for the model in the prediction directory.
# This assumes filenames match the pattern: ttd_{patient_id}_pred_seg.nii.gz
pred_images = sorted([f for f in os.listdir(pred_path) if re.match(rf'{model}_\d{{5}}_pred_seg.nii.gz', f)])
pred_patient_ids = [re.search(rf'{model}_(\d{{5}})_pred_seg', f).group(1) for f in pred_images]

# Define the segmentation labels (0=BG, 1=NCR, 2=ED, 3=ET)
labels = [0, 1, 2, 3]

# Initialize an aggregated confusion matrix
aggregated_cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

# Iterate over each patient in the test set
for pid in pred_patient_ids:
    # Build ground truth path (assumes folder: BraTS2021_{pid} and filename: BraTS2021_{pid}_seg.nii.gz)
    gt_seg_path = os.path.join(gt_path, f"BraTS2021_{pid}", f"BraTS2021_{pid}_seg.nii.gz")
    if not os.path.exists(gt_seg_path):
        print(f"Ground truth segmentation not found for patient {pid}. Skipping.")
        continue
    gt_seg = nib.load(gt_seg_path).get_fdata().astype(np.int32)

    # Build prediction path (filename: ttd_{pid}_pred_seg.nii.gz)
    pred_seg_file = f"{model}_{pid}_pred_seg.nii.gz"
    pred_seg_path = os.path.join(pred_path, pred_seg_file)
    if not os.path.exists(pred_seg_path):
        print(f"Prediction segmentation not found for patient {pid}. Skipping.")
        continue
    pred_seg = nib.load(pred_seg_path).get_fdata().astype(np.int32)

    # Flatten the 3D volumes to 1D arrays of voxel labels
    y_true = gt_seg.flatten()
    y_pred = pred_seg.flatten()

    # Compute the confusion matrix for this patient
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Sum into the aggregated confusion matrix
    aggregated_cm += cm

# Normalize the aggregated confusion matrix by rows (so each true-class sums to 1)
cm_normalized = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1, keepdims=True)
# Replace any NaN (from divisions by zero) with 0
cm_normalized = np.nan_to_num(cm_normalized)
# For annotation, convert normalized values to percentages (rounded to one decimal place)
cm_percent = (cm_normalized * 100).round(1)

# Create a custom colormap (this example uses a gradient from light blue to dark purple)
colors = ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#6e016b']
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom', colors)

# Plot the normalized aggregated confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=custom_cmap,
            xticklabels=["BG", "NCR", "ED", "ET"],
            yticklabels=["BG", "NCR", "ED", "ET"],
            vmin=0, vmax=100)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (%)")
plt.savefig(f"./confusion_matrices/{model}_confusion_matrix.png")
plt.show()
