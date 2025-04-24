import os 
import subprocess
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths for different data sources
# ood_data_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\braintumor_data" # Path to OOD cases
ood_data_path = "/home/magata/data/braintumor_data"

def get_ood_cases():
    """Get list of out-of-distribution case folders."""
    if not os.path.exists(ood_data_path):
        raise FileNotFoundError(f"OOD data directory not found at {ood_data_path}")
    return [os.path.join(ood_data_path, f, "original") for f in os.listdir(ood_data_path) 
            if os.path.isdir(os.path.join(ood_data_path, f))]

def skull_strip_with_hd_bet(input_path, output_dir):
    """
    Run HD-BET skull stripping on a single NIfTI file.
    - input_path : full path to input volume (e.g. flair.nii.gz)
    - output_dir : directory in which to write results
    Returns path to the skull‑stripped image (.nii.gz).
    """
    import subprocess, os

    # make sure the folder exists
    os.makedirs(output_dir, exist_ok=True)

    # extract bare basename (no .nii or .nii.gz)
    base = os.path.basename(input_path)
    if base.endswith('.nii.gz'):
        base = base[:-7]
    elif base.endswith('.nii'):
        base = base[:-4]

    # build the two output files
    skull_file = os.path.join(output_dir, f"{base}_bet.nii.gz")
    mask_file  = os.path.join(output_dir, f"{base}_mask.nii.gz")

    # call hd-bet, outputting a single file;
    # --save_mask will also emit the mask_file
    subprocess.run([
        "hd-bet",
        "-i", input_path,
        "-o", skull_file,
        "-device", "cuda",
        "--disable_tta",
        "--save_bet_mask"
    ], check=True)

    # sanity‑check
    if not os.path.exists(skull_file):
        raise FileNotFoundError(f"HD-BET failed to write stripped image at {skull_file}")
    # if you care about the mask, you can also test mask_file here

    return skull_file



def preprocess_images(nifti_paths, output_dir, progress_callback=None):
    """
    Skull-strips and realigns all images in 'nifti_paths' to the first one (as reference).
    The preprocessed images are saved into the specified output_dir.
    Optionally updates a Streamlit progress bar.
    """
    if not nifti_paths:
        return []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    stripped_paths = []
    total_steps = len(nifti_paths) * 2  # Skull stripping + registration
    current_step = 0

    for p in nifti_paths:
        base_name = os.path.basename(p).replace(".nii", "").replace(".gz", "")
        # choose a _directory_ to hold HD‑BET's outputs for this volume
        this_out_dir = os.path.join(output_dir, base_name + "_hd-bet")
        stripped = skull_strip_with_hd_bet(p, output_dir=this_out_dir)

        stripped_paths.append(stripped)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    reference_path = stripped_paths[0]
    ref_img_sitk = sitk.ReadImage(reference_path, sitk.sitkFloat32)

    output_paths = []
    ref_output_path = os.path.join(output_dir, f"preproc_{os.path.basename(reference_path)}")
    sitk.WriteImage(ref_img_sitk, ref_output_path)
    output_paths.append(ref_output_path)

    for path in stripped_paths[1:]:
        mov_img_sitk = sitk.ReadImage(path, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            ref_img_sitk,
            mov_img_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=40)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        final_transform = registration_method.Execute(
            sitk.Cast(ref_img_sitk, sitk.sitkFloat32),
            sitk.Cast(mov_img_sitk, sitk.sitkFloat32),
        )

        aligned_img = sitk.Resample(
            mov_img_sitk,
            ref_img_sitk,
            final_transform,
            sitk.sitkLinear,
            0.0,
            mov_img_sitk.GetPixelID(),
        )

        out_path = os.path.join(output_dir, f"preproc_{os.path.basename(path)}")
        sitk.WriteImage(aligned_img, out_path)
        output_paths.append(out_path)

        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    return output_paths

def process_ood_case(patient_folder):
    """
    Process a single out-of-distribution case.
    """
    print(f"Processing OOD case: {patient_folder}")
    
    # Create preprocessed directory
    preprocessed_dir = os.path.join(patient_folder, "preprocessed1")
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
    preprocessed_paths = preprocess_images(nifti_files, preprocessed_dir)
    
    # Visualize the results
    visualize_preprocessed_images(preprocessed_paths)

def visualize_preprocessed_images(preprocessed_paths):
    """
    Visualize the preprocessed images.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, path in enumerate(preprocessed_paths[:4]):  # Show up to 4 images
        img = nib.load(path).get_fdata()
        center_slice = img.shape[2] // 2
        ax = axs[i//2, i%2]
        ax.imshow(img[:, :, center_slice], cmap='gray')
        ax.set_title(os.path.basename(path))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Process all OOD cases
    ood_cases = get_ood_cases()
    for case in ood_cases:
        print(case)
        if case != "/home/magata/data/braintumor_data/VIGO_01/original":
            continue
        process_ood_case(case)

if __name__ == "__main__":
    main()
    
