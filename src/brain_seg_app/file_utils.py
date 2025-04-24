import os, glob, re
import numpy as np
import nibabel as nib
import pydicom
import streamlit as st
import tempfile


def check_file_format(file):
    """
    Checks if a file is DICOM or NIfTI based on its name and/or a lightweight DICOM read.

    Parameters:
    -----------
    file : UploadedFile
        The file object from Streamlit.

    Returns:
    --------
    str or None
        "nifti" if the file is recognized as NIfTI,
        "dicom" if recognized as DICOM,
        or None otherwise.
    """
    try:
        file.seek(0)
        if file.name.endswith(".nii") or file.name.endswith(".nii.gz"):
            return "nifti"
        elif file.name.endswith(".dcm") or pydicom.dcmread(
            file, stop_before_pixels=True
        ):
            return "dicom"
    except Exception as e:
        st.error(f"Error checking format of {file.name}: {e}")
    return None

def convert_dicom_to_nifti(dicom_files, output_filename="converted.nii"):
    """
    Converts a list of DICOM files (dicom_files) into a single NIfTI volume (output_filename).
    Assumes axial slices and uses a simple diagonal affine.

    Parameters:
    -----------
    dicom_files : list
        List of file-like objects for DICOM slices.
    output_filename : str
        Output path for the resulting NIfTI file.

    Returns:
    --------
    str or None
        The path to the created NIfTI file, or None if an error occurred.
    """
    try:
        # Ensure each file's pointer is at the start
        for f in dicom_files:
            f.seek(0)

        # Read all DICOM slices
        dicom_slices = [pydicom.dcmread(f) for f in dicom_files]
        # Sort by ImagePositionPatient (Z-direction)
        dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        # Build a 3D volume
        image_array = np.stack([s.pixel_array for s in dicom_slices], axis=0)

        # Retrieve pixel spacing and slice thickness
        slice_thickness = float(dicom_slices[0].SliceThickness)
        pixel_spacing = [float(i) for i in dicom_slices[0].PixelSpacing]

        # Build a simple diagonal affine
        affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])

        # Save as NIfTI
        nifti_img = nib.Nifti1Image(image_array, affine)
        output_path = os.path.join(os.getcwd(), output_filename)
        nib.save(nifti_img, output_path)
        return output_path

    except Exception as e:
        st.error(f"Error converting DICOM to NIfTI: {e}")
        return None
    
def reorder_modalities(nifti_files):
    """
    Reorders the given list of 4 NIfTI file paths/names to match the desired order:
    [Flair, T1ce, T1, T2].
    We do partial matching on the filename to detect each modality.
    If a match is missing or ambiguous, we warn the user and place it last.
    """
    # Desired order and the corresponding keywords to look for in filenames
    desired_order = [
        ("Flair", ["flair"]),
        ("T1ce",  ["t1ce", "t1c"]),
        ("T1",    ["t1"]),
        ("T2",    ["t2"])
    ]
    
    # We'll store matched files in a dict: modality_name -> file_path
    matched = {}
    unmatched_files = []

    for file_path in nifti_files:
        fname_lower = file_path.lower()
        found = False
        # Try to match each desired modality in order
        for modality_name, keywords in desired_order:
            # If we've already matched something for that modality, skip it
            if modality_name in matched:
                continue
            # Check if any keyword is in the filename
            if any(kw in fname_lower for kw in keywords):
                matched[modality_name] = file_path
                found = True
                break
        if not found:
            unmatched_files.append(file_path)
    
    # Build final list in the desired order
    final_list = []
    for modality_name, _ in desired_order:
        if modality_name in matched:
            final_list.append(matched[modality_name])
        else:
            # If missing, skip or fill with unmatched if available
            if unmatched_files:
                missing_file = unmatched_files.pop(0)
                final_list.append(missing_file)
                st.warning(f"Missing {modality_name} in filenames. Using '{missing_file}' as fallback.")
            else:
                st.warning(f"Missing {modality_name} and no fallback available.")
                # Insert a placeholder or skip
                final_list.append("")
    
    # If there are still leftover unmatched files, append them at the end
    for leftover in unmatched_files:
        st.warning(f"Unmatched file '{leftover}' placed at end.")
        final_list.append(leftover)
    
    return final_list

def find_available_outputs(output_dir="./assets/segmentations"):
    """
    Returns a dictionary mapping each patient_id to the paths of:
      - 'seg': segmentation file
      - 'uncertainty_NCR': path to the NCR uncertainty map
      - 'uncertainty_ED':  path to the ED uncertainty map
      - 'uncertainty_ET':  path to the ET uncertainty map
    """
    seg_files = glob.glob(os.path.join(output_dir, "segmentation_*.nii.gz"))
    ncr_files = glob.glob(os.path.join(output_dir, "uncertainty_NCR_*.nii.gz"))
    ed_files  = glob.glob(os.path.join(output_dir, "uncertainty_ED_*.nii.gz"))
    et_files  = glob.glob(os.path.join(output_dir, "uncertainty_ET_*.nii.gz"))
    glob_files  = glob.glob(os.path.join(output_dir, "uncertainty_global_*.nii.gz"))
    softmax_files = glob.glob(os.path.join(output_dir, "softmax_*.nii.gz"))
    gt_files = glob.glob(os.path.join(output_dir, "*seg*.nii.gz"))
   
    outputs = {}

    def extract_id(path):
        """
        Extracts the patient ID from a filename by capturing tokens that consist solely
        of digits or uppercase letters, while ignoring tokens that are exactly "NCR", "ED", or "ET".
        
        Examples:
        "preproc_ARE_flair.nii"         -> returns "ARE"
        "uncertainty_NCR_00657.nii.gz"    -> returns "00657"
        "preproc_NCR_ABC.nii"             -> returns "ABC"
        """
        fname = os.path.basename(path)
        # Capture tokens that follow an underscore and consist of digits and/or uppercase letters.
        tokens = re.findall(r'_([\dA-Z]+)', fname)
        
        # Define tokens to ignore.
        ignore = {"NCR", "ED", "ET"}
        
        # Filter tokens: Allow numeric tokens or alphabetic tokens not in ignore.
        valid_tokens = []
        for token in tokens:
            if token.isdigit():
                valid_tokens.append(token)
            elif token.isalpha() and token not in ignore:
                valid_tokens.append(token)
        
        # Return the last valid token if available (you could also choose the first, depending on your naming scheme).
        if valid_tokens:
            return valid_tokens[-1]
        return "UNKNOWN"

    # Fill the dictionary
    for f in seg_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["seg"] = f
    for f in ncr_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["uncertainty_NCR"] = f
    for f in ed_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["uncertainty_ED"] = f
    for f in et_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["uncertainty_ET"] = f
    for f in glob_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["uncertainty_global"] = f
    for f in softmax_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["softmax"] = f
    for f in gt_files:
        pid = extract_id(f)
        outputs.setdefault(pid, {})["gt"] = f

    return outputs

def nifti_to_bytes(nii_img):
    """
    Save a NIfTI image to a temporary file and return its contents as bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        nib.save(nii_img, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        os.remove(tmp_path)
    return data
