import os
import nibabel as nib
import numpy as np
import nibabel.processing as nibproc

# Define your data paths
ood_data_path = "/home/magata/data/braintumor_data/VIGO_01"
preprocessed_path = os.path.join(ood_data_path, "original", "preprocessed1")

def load_preprocessed_image(preprocessed_path, prefix):
    """Load a preprocessed image that starts with the given prefix."""
    for f in os.listdir(preprocessed_path):
        if f.startswith(prefix):
            return nib.load(os.path.join(preprocessed_path, f))
    return None

# Load a reference image from your preprocessed images (for example, a FLAIR skullstripped image)
ref_prefix = "preproc_VIGO_01_11102018_FLAIR_orig_bet.nii.gz"
ref_img = load_preprocessed_image(preprocessed_path, ref_prefix)
if ref_img is None:
    raise FileNotFoundError("Could not find preprocessed reference image")

origin = ood_data_path
flair_file = None
t1gad_file = None
# Find VOI files for FLAIR and T1GAD
for f in os.listdir(origin):
    # skip ADS pseudo-files
    if ':' in f:
        continue

    # only care about .nii or .nii.gz
    if not (f.lower().endswith('.nii') or f.lower().endswith('.nii.gz')):
        continue

    # pick out your VOIs
    if "VOIFLAIR" in f:
        flair_file = os.path.join(origin, f)
    elif "VOIT1GAD" in f:
        t1gad_file = os.path.join(origin, f)

if not (flair_file and t1gad_file):
    raise FileNotFoundError("Could not find both VOIFLAIR and VOIT1GAD files")

# Load segmentation images with nibabel
flair_img = nib.load(flair_file)
t1gad_img = nib.load(t1gad_file)

# Resample the segmentation images to the reference image’s grid using header information only.
# Using order=0 for nearest neighbor ensures that label values remain intact.
flair_img_resampled = nibproc.resample_from_to(flair_img, (ref_img.shape, ref_img.affine), order=0)
t1gad_img_resampled = nibproc.resample_from_to(t1gad_img, (ref_img.shape, ref_img.affine), order=0)

# Get the data arrays from the resampled images
flair_data = flair_img_resampled.get_fdata()
t1gad_data = t1gad_img_resampled.get_fdata()

print("Resampled FLAIR shape:", flair_data.shape)
print("Resampled T1GAD shape:", t1gad_data.shape)

# Combine segmentations
# For example, assign label 2 for FLAIR (ED) and label 3 for T1GAD (ET)
combined_seg = np.zeros_like(flair_data)
combined_seg[flair_data > 0] = 2
combined_seg[t1gad_data > 0] = 3

# Save the combined segmentation using the reference image's affine and header
combined_img = nib.Nifti1Image(combined_seg, ref_img.affine, ref_img.header)
output_path = os.path.join(ood_data_path, "combined_seg.nii.gz")
nib.save(combined_img, output_path)
print("Saved combined segmentation to", output_path)
