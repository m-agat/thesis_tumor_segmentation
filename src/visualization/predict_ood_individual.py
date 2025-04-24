import os
import sys
import torch
import numpy as np
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
import re
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import pandas as pd
sys.path.append("../")
import models.models as models
import config.config as config
from dataset import dataloaders
from dataset.transforms import get_test_transforms
from monai import data

# Path to OOD cases
ood_data_path = "/home/magata/data/braintumor_data"

def get_preprocessed_files(directory):
    """
    Get list of preprocessed files from a directory.
    Only returns files that have 'preproc_' in their name.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and 'preproc_' in f]
    
    if not files:
        print(f"Warning: No preprocessed files found in directory: {directory}")
        return []
        
    return files

def get_ood_cases():
    """Get list of out-of-distribution case folders."""
    if not os.path.exists(ood_data_path):
        raise FileNotFoundError(f"OOD data directory not found at {ood_data_path}")
    return [os.path.join(ood_data_path, f, "original", "preprocessed1") for f in os.listdir(ood_data_path) 
            if os.path.isdir(os.path.join(ood_data_path, f))]

def create_data_loader(mri_scans):
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
    dummy_label_array = np.zeros(shape, dtype=np.uint8)
    dummy_label_path = os.path.join(os.getcwd(), "dummy_label.nii.gz")
    dummy_label_nii = nib.Nifti1Image(dummy_label_array, affine=first_img.affine)
    nib.save(dummy_label_nii, dummy_label_path)

    entry = {
        "image": mri_scans,       # list of file paths, one per modality
        "label": dummy_label_path,     # dummy label
        "path": mri_scans[0]      # use the first modality's path as reference
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

def load_model(model_class, checkpoint_path, device):
    """
    Load a segmentation model from a checkpoint.
    """
    model = model_class
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, partial(
        sliding_window_inference,
        roi_size=config.roi,
        sw_batch_size=config.sw_batch_size,
        predictor=model,
        overlap=config.infer_overlap,
    )

def save_segmentation_as_nifti(predicted_segmentation, reference_image_path, output_path):
    """
    Save the predicted segmentation as a NIfTI file.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()

    predicted_segmentation = predicted_segmentation.astype(np.uint8)

    ref_img = nib.load(reference_image_path)
    seg_img = nib.Nifti1Image(predicted_segmentation, affine=ref_img.affine, header=ref_img.header)
    nib.save(seg_img, output_path)

    print(f"Segmentation saved to {output_path}")

def predict_ood_case(model_name, case_path, output_dir):
    """
    Predict segmentation for a single OOD case using a specific model.
    
    Args:
        model_name: Name of the model to use (e.g., "attunet", "swinunetr", "segresnet")
        case_path: Path to the OOD case directory
        output_dir: Directory to save the predictions
    """
    # Map model names to their corresponding model class and checkpoint path
    model_config = {
        "attunet": (models.attunet_model, config.model_paths["attunet"]),
        "swinunetr": (models.swinunetr_model, config.model_paths["swinunetr"]),
        "segresnet": (models.segresnet_model, config.model_paths["segresnet"]),
        "vnet": (models.vnet_model, config.model_paths["vnet"]),
    }

    if model_name not in model_config:
        raise ValueError(f"Selected model '{model_name}' is not recognized. Choose from: {list(model_config.keys())}")

    # Load the model
    model_class, checkpoint_path = model_config[model_name]
    model, inferer = load_model(model_class, checkpoint_path, config.device)

    # Get preprocessed files for the case
    preprocessed_files = get_preprocessed_files(case_path)
    if not preprocessed_files:
        print(f"No preprocessed files found for case: {case_path}")
        return

    # Create data loader
    data_loader = create_data_loader(preprocessed_files)

    # Create output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Extract patient ID from case path
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(preprocessed_files[0]))))
    print(f"\nProcessing patient: {patient_id} using {model_name}\n")

    # Perform prediction
    with torch.no_grad():
        for batch_data in data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]

            logits = inferer(image).squeeze(0)
            seg = torch.nn.functional.softmax(logits, dim=0).argmax(dim=0).unsqueeze(0)
            seg = seg.squeeze(0)

            output_path = os.path.join(model_output_dir, f"{model_name}_{patient_id}_pred_seg.nii.gz")
            save_segmentation_as_nifti(seg, reference_image_path, output_path)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Get all OOD cases
    ood_cases = get_ood_cases()
    print("Found OOD cases:", ood_cases)
    
    # Define which cases to process
    target_cases = ["VIGO_01", "VIGO_03"]
    
    # Define which models to use
    models_to_use = ["attunet", "swinunetr", "segresnet", "vnet"]
    
    # Create output directory
    output_dir = os.path.join("segmentations", "individual_models")
    os.makedirs(output_dir, exist_ok=True)

    # Process each case
    for case in ood_cases:
        # Get preprocessed files first
        preprocessed_files = get_preprocessed_files(case)
        if not preprocessed_files:
            print(f"Skipping case {case} - no preprocessed files found")
            continue
            
        # Extract patient ID from case path
        try:
            patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(preprocessed_files[0]))))
            print(f"Processing case: {patient_id}")
        except IndexError:
            print(f"Error extracting patient ID from case: {case}")
            continue
            
        # Skip if not in target cases
        if patient_id not in target_cases:
            print(f"Skipping case {patient_id} - not in target cases")
            continue
            
        # Process with each model
        for model_name in models_to_use:
            print(f"Using model: {model_name}")
            try:
                predict_ood_case(model_name, case, output_dir)
            except Exception as e:
                print(f"Error processing case {patient_id} with model {model_name}: {str(e)}")
                continue 