import numpy as np
from scipy.ndimage import shift


def combine_masks(mask1, mask2):
    """
    Combines two 3D masks into one to simulate a case with two tumors.
    """
    combined_mask = np.zeros_like(mask1)

    # Priority: Combine tumor labels (assuming labels: 0 = background, 1 = NCR, 2 = ED, 4 = ET)
    combined_mask[mask1 == 1] = 1  # NCR from mask1
    combined_mask[mask2 == 1] = 1  # NCR from mask2

    combined_mask[mask1 == 2] = 2  # ED from mask1
    combined_mask[mask2 == 2] = 2  # ED from mask2

    combined_mask[mask1 == 4] = 4  # ET from mask1
    combined_mask[mask2 == 4] = 4  # ET from mask2

    return combined_mask


def combine_uncertainty_maps(uncertainty1, uncertainty2, method="max"):
    """
    Combines two 3D uncertainty maps.

    Args:
        uncertainty1: numpy array of the first uncertainty map (floating-point).
        uncertainty2: numpy array of the second uncertainty map (floating-point).
        method: 'max' for max uncertainty, 'mean' for averaging, or 'sum' for summing.

    Returns:
        Combined uncertainty map.
    """
    if method == "max":
        combined_uncertainty = np.maximum(uncertainty1, uncertainty2)
    elif method == "mean":
        combined_uncertainty = (uncertainty1 + uncertainty2) / 2
    elif method == "sum":
        combined_uncertainty = uncertainty1 + uncertainty2
    else:
        raise ValueError("Invalid combination method. Choose 'max', 'mean', or 'sum'.")

    return combined_uncertainty


def combine_brain_scans(scan1, scan2, method="mean"):
    """
    Combines two brain MRI scans after normalization and resampling.
    """

    if method == "mean":
        combined_scan = (scan1 + scan2) / 2
    elif method == "max":
        combined_scan = np.maximum(scan1, scan2)
    elif method == "sum":
        combined_scan = scan1 + scan2
    else:
        raise ValueError("Invalid method. Choose 'mean', 'max', or 'sum'.")

    return combined_scan


# Load mask volumes (assuming NIfTI format)
import nibabel as nib

# mask1_nifti = nib.load('/home/magata/data/brats2021challenge/split/test/BraTS2021_01309/BraTS2021_01309_seg.nii.gz')
# mask2_nifti = nib.load('/home/magata/data/brats2021challenge/split/test/BraTS2021_00688/BraTS2021_00688_seg.nii.gz')

# mask1 = mask1_nifti.get_fdata().astype(np.int32)
# mask2 = mask2_nifti.get_fdata().astype(np.int32)

# # Combine masks
# combined_mask = combine_masks(mask1, mask2)

# # Save the combined mask
# combined_nifti = nib.Nifti1Image(combined_mask, affine=mask1_nifti.affine)
# nib.save(combined_nifti, '/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/other/brats_example_segmentations/BraTS2021_00688_AND_01309_seg.nii.gz')

# Load uncertainty maps (NIfTI format)
# uncertainty1_nifti = nib.load('/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/uncertainty/outputs/BraTS2021_01309_uncertainty_map_ET.nii.gz')
# uncertainty2_nifti = nib.load('/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/uncertainty/outputs/BraTS2021_00688_uncertainty_map_ET.nii.gz')

# uncertainty1 = uncertainty1_nifti.get_fdata().astype(np.float32)  # Ensure floating-point type
# uncertainty2 = uncertainty2_nifti.get_fdata().astype(np.float32)

# # Combine the uncertainty maps (e.g., using max)
# combined_uncertainty = combine_uncertainty_maps(uncertainty1, uncertainty2, method='max')

# # Save the combined uncertainty map as a NIfTI file
# combined_nifti = nib.Nifti1Image(combined_uncertainty, affine=uncertainty1_nifti.affine)
# nib.save(combined_nifti, '/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/other/brats_example_segmentations/BraTS2021_00688_AND_01309_uncertainty_map_ET.nii.gz')

# Load the NIfTI brain scans
scan1_nifti = nib.load(
    "/home/magata/data/brats2021challenge/split/test/BraTS2021_00688/BraTS2021_00688_flair.nii.gz"
)
scan2_nifti = nib.load(
    "/home/magata/data/brats2021challenge/split/test/BraTS2021_01309/BraTS2021_01309_flair.nii.gz"
)

scan1 = scan1_nifti.get_fdata().astype(np.float32)
scan2 = scan2_nifti.get_fdata().astype(np.float32)

# Combine the brain scans
combined_scan = combine_brain_scans(scan1, scan2, method="max")

# Save the combined scan
combined_nifti = nib.Nifti1Image(combined_scan, affine=scan1_nifti.affine)
nib.save(
    combined_nifti,
    "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/other/brats_example_segmentations/BraTS2021_00688_AND_01309_flair.nii.gz",
)
