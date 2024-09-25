import os
import nibabel as nib
import numpy as np
import pandas as pd 
from scipy.ndimage import zoom

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


def csv_to_dict(csv_file_path, key_column, value_column, convert_numeric=True, dropna=True):
    """
    Convert a CSV file into a dictionary, where the user specifies the key and value columns.

    Parameters:
    - csv_file_path (str): Path to the CSV file.
    - key_column (str): The column to be used as the keys of the dictionary.
    - value_column (str): The column to be used as the values of the dictionary.
    - convert_numeric (bool): If True, attempts to convert the values to numeric types (e.g., int, float).
    - dropna (bool): If True, drops rows with missing values in the key or value columns.

    Returns:
    - dict: A dictionary where keys are from the `key_column` and values from `value_column`.
    """
    df = pd.read_csv(csv_file_path)

    if dropna:
        df = df.dropna(subset=[key_column, value_column])

    if convert_numeric:
        df[value_column] = pd.to_numeric(df[value_column], errors='ignore')

    return dict(zip(df[key_column], df[value_column]))

def resample_to_shape(pred, target_shape):
    """
    Resample the prediction to match the target shape using nearest-neighbor interpolation.
    """
    factors = [t / s for t, s in zip(target_shape, pred.shape)]  # Calculate resampling factors
    resampled_pred = zoom(pred, factors, order=0)  # Nearest neighbor interpolation (order=0)
    return resampled_pred