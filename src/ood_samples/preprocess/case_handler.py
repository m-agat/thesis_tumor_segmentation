import os
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import get_ood_cases
from transforms import preprocess_images

def process_ood_case(patient_folder):
    """
    Process a single out-of-distribution case.
    """
    print(f"Processing OOD case: {patient_folder}")
    
    # Create preprocessed directory
    preprocessed_dir = os.path.join(patient_folder, "preprocessed_scans")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Get all NIfTI files in the patient folder
    nifti_files = []
    for root, _, files in os.walk(patient_folder):
        for file in files:
            # Skip the compressed folders, only use the .nii files
            if file.endswith('.nii') and not file.endswith('.nii.gz'):
                nifti_files.append(os.path.join(root, file))
    
    if not nifti_files:
        print(f"No NIfTI files found in {patient_folder}")
        return
    
    # Sort files in the required order: FLAIR, T1GAD, T1, T2
    def get_modality_order(file_path):
        """Sort files in the required order: FLAIR, T1GAD, T1, T2"""
        filename = os.path.basename(file_path).lower()
        if 'flair_orig' in filename:
            return 0
        elif 't1gad_orig' in filename:
            return 1
        elif 't1_orig' in filename and 't1gad' not in filename:
            return 2
        elif 't2_orig' in filename:
            return 3
        else:
            return 4  # Put unknown modalities at the end
    
    nifti_files.sort(key=get_modality_order)
    
    # Verify we have all required modalities
    required_modalities = ['flair_orig', 't1gad_orig', 't1_orig', 't2_orig']
    found_modalities = [os.path.basename(f).lower() for f in nifti_files]
    missing_modalities = [m for m in required_modalities if not any(m in f for f in found_modalities)]
    
    if missing_modalities:
        print(f"Warning: Missing required modalities in {patient_folder}: {', '.join(missing_modalities)}")
        return
    
    # Preprocess the images
    preprocess_images(nifti_files, preprocessed_dir)
    print(f"[INFO] Preprocessed scans saved to: {preprocessed_dir} ")