import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys 
from monai import data
import re 

sys.path.append("../")
import models as models
import ensemble.simple_averaging as simple_avg
import ensemble.hybrid_uncertainty_weighting as hyb_unc
import ensemble.performance_weighted as pwe
import ensemble.tta_only_weighted as tta_pred
import ensemble.ttd_only_weighted as ttd_pred
import config.config as config
from dataset import dataloaders
from dataset.transforms import get_test_transforms

# Path to OOD cases
# ood_data_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\braintumor_data" 
ood_data_path = "/home/magata/data/braintumor_data"

def get_preprocessed_files(directory):
    """
    Get list of preprocessed files from a directory.
    Only returns files that have 'preproc_' in their name.
    
    Args:
        directory (str): Path to directory containing preprocessed files
        
    Returns:
        list: List of full paths to preprocessed files
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and 'preproc_' in f]

def get_ood_cases():
    """Get list of out-of-distribution case folders."""
    if not os.path.exists(ood_data_path):
        raise FileNotFoundError(f"OOD data directory not found at {ood_data_path}")
    return [os.path.join(ood_data_path, f, "original", "preprocessed1") for f in os.listdir(ood_data_path) 
            if os.path.isdir(os.path.join(ood_data_path, f))]

def load_predict_models():
    # loads attunet, segresnet, swinunetr
    return hyb_unc.load_all_models()

def create_data_loader(mri_scans, pid):
    """
    Constructs a MONAI DataLoader from a list of MRI scan file paths.
    Here, mri_scans is a list containing file paths for each modality.
    The modalities are ordered as: FLAIR, T1CE, T1, T2.
    """
    # Define the order of modalities
    modality_order = {
        'flair': 0,
        't1ce': 1,
        't1': 2,
        't2': 3
    }
    
    # Sort files based on modality
    mri_scans = sorted(mri_scans, key=lambda x: modality_order.get(
        next((mod for mod in modality_order.keys() if mod in x.lower()), 4)
    ))
    
    print("Loaded MRI scans in order:", [os.path.basename(f) for f in mri_scans])

    first_img = nib.load(mri_scans[0])
    shape = first_img.get_fdata().shape  
    
    # Create a dummy label with the same spatial dimensions as shape var
    # dummy_label_array = np.zeros(shape, dtype=np.uint8)
    # dummy_label_path = os.path.join(os.getcwd(), "dummy_label.nii.gz")
    # dummy_label_nii = nib.Nifti1Image(dummy_label_array, affine=first_img.affine)
    # nib.save(dummy_label_nii, dummy_label_path)

    gt_path = ood_data_path + f"/{pid}/{pid}_seg.nii.gz"

    entry = {
        "image": mri_scans,       # list of file paths, one per modality
        "label": gt_path,     # dummy label
        "path": mri_scans[0]      # use the first modality's path as reference (e.g., for patient ID)
    }
    test_files = [entry]

    test_transform = get_test_transforms()

    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return test_loader

if __name__ == "__main__":
    ood_cases = get_ood_cases()
    models = load_predict_models()

    # Predict segmentation for each case
    for case in ood_cases:
        preprocessed_files = get_preprocessed_files(case)

        # Skip if no preprocessed files found
        if not preprocessed_files:
            print(f"No preprocessed files found for case: {case}")
            continue

        # Extract patient ID from case path
        # Get patient ID by going up two directories from the preprocessed file path
        # preprocessed_files[0] path is like: /path/to/VIGO_01/original/preprocessed1/file.nii.gz
        # We want to extract "VIGO_01"
        patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(preprocessed_files[0]))))
        print("Patient ID: ", patient_id)

        # if patient_id == "VIGO_02" or patient_id == "VIGO_03" or patient_id == "MMPG":
        if patient_id == "VIGO_01" or patient_id == "VIGO_03":
            print("creating data loader")
            data_loader = create_data_loader(preprocessed_files, pid=patient_id)

            # Create separate output directories for each method
            simple_avg_dir = os.path.join("segmentations", "simple_avg")
            pwe_dir = os.path.join("segmentations", "pwe") 
            tta_dir = os.path.join("segmentations", "tta")
            ttd_dir = os.path.join("segmentations", "ttd")
            hybrid_dir = os.path.join("segmentations", "hybrid")

            # Create all directories
            for directory in [simple_avg_dir, pwe_dir, tta_dir, ttd_dir, hybrid_dir]:
                os.makedirs(directory, exist_ok=True)
            # Create a directory for segmentation outputs
            seg_output_dir = os.path.join("segmentations")
            os.makedirs(seg_output_dir, exist_ok=True)

            # Predict segmentation
            print("predicting segmentation")
            simple_avg.ensemble_segmentation(data_loader, models, output_dir=simple_avg_dir, ood=True)
            pwe.ensemble_segmentation(data_loader, models, wts={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}, out_dir=pwe_dir, ood=True)
            # tta_pred.ensemble_segmentation(data_loader, models, wts={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}, n_iter=10, out_dir=tta_dir)
            # ttd_pred.ensemble_segmentation(data_loader, models, wts={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}, n_iter=10, out_dir=ttd_dir)
            # hyb_unc.ensemble_segmentation(data_loader, models, wts={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}, n_iter=10, out_dir=hybrid_dir)
            print("Done")









