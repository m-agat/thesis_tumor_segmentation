import os
import nibabel as nib
import numpy as np

def create_case_folder(results_dir, case_num):
    """
    Create a directory to store segmentations.
    """
    case_folder = os.path.join(results_dir, "segmentations", f"BraTS2021_{case_num}")
    if not os.path.exists(case_folder):
        os.makedirs(case_folder)
    return case_folder

def save_segmentation_as_nifti(segmentation, reference_nifti_path, output_path):
    """
    Save segmentation as a NIfTI file using the affine matrix of the reference image.
    """
    # Load the original image to copy affine information (geometric orientation)
    reference_img = nib.load(reference_nifti_path)
    affine = reference_img.affine  # Get affine for saving with correct orientation

    # Ensure segmentation has the correct shape
    if segmentation.shape != reference_img.shape:
        raise ValueError(f"Segmentation shape {segmentation.shape} does not match reference image shape {reference_img.shape}")

    # Create a new NIfTI image with the segmentation and affine
    nifti_img = nib.Nifti1Image(segmentation.astype(np.int16), affine)

    # Save to the specified output path
    nib.save(nifti_img, output_path)
    print(f"Segmentation saved to {output_path}")
