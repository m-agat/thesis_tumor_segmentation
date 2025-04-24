import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the softmax probability map
softmax_path = "./assets/segmentations/softmax_00025.nii.gz"
# softmax_path = "../ensemble/output/hybrid/softmax_01556.nii.gz"
# softmax_path = "../ensemble/output_segmentations/perf_weight/perf_weight_softmax_00285.nii.gz"
segmentation_path = "./assets/segmentations/segmentation_00025.nii.gz"

# load softmax probs and the fused hard segmentation
softmax = nib.load(softmax_path).get_fdata()  # shape (4, H, W, D)
seg = nib.load(segmentation_path).get_fdata()  # shape (H, W, D), ints 0–3

regions = {
    "NCR": 1,
    "ED":  2,
    "ET":  3,
}

# Print unique values in softmax probabilities for each region
print("\nUnique softmax probabilities per region:")
for i, region in enumerate(["BG", "NCR", "ED", "ET"]):
    unique_vals = np.unique(softmax[i])
    print(f"\n{region}:")
    print(f"Number of unique values: {len(unique_vals)}")
    print(f"Min: {unique_vals.min():.6f}")
    print(f"Max: {unique_vals.max():.6f}")
    print(f"Sample of values: {unique_vals[:5]}...")

plt.figure(figsize=(15,5))
for i, (name, lbl) in enumerate(regions.items()):
    probs = softmax[lbl]             # raw probability map for this region
    mask  = (seg == lbl)             # only voxels predicted as that label
    if mask.sum() == 0:
        print(f"No voxels predicted as {name}")
        continue

    vals = probs[mask]               # only histogram within the mask
    plt.subplot(1,3,i+1)
    plt.hist(vals, bins=50, range=(0,1))
    plt.title(f"{name} probability (within mask)")
    plt.xlabel("p")
    plt.ylabel("voxel count")
plt.tight_layout()
plt.show()
