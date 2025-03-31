import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os
import torch
from monai import data
import glob
import plotly.graph_objects as go
import subprocess
import re 
import tempfile 

# Ensemble model
import final_ensemble_model as fem
from dataset.transforms import get_test_transforms
import config.config as config 

# Optional: SimpleITK for registration and intensity rescaling
import SimpleITK as sitk

print(os.getcwd())

# ------------------------------------------------------------------------------
# Cache the model loading so that models are loaded once per session.
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ensemble_models():
    return fem.load_all_models()

# ------------------------------------------------------------------------------
# Create a simple test loader from a NIfTI file.
# ------------------------------------------------------------------------------
def create_test_loader(mri_scans):
    """
    Constructs a MONAI DataLoader from a list of MRI scan file paths.
    Here, mri_scans is a list containing file paths for each modality.
    """
    print(mri_scans)
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


# --------------------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tumor Segmentation", layout="wide", initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_slice_lazy(path, slice_idx):
    img = nib.load(path)
    return img.dataobj[..., slice_idx]

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

def show_simple(image_3d, slice_idx, title=""):
    """
    Displays a single slice from a 3D volume using Matplotlib.
    image_3d should have shape [D, H, W].

    Parameters:
    -----------
    image_3d : np.ndarray
        3D array representing the volume.
    slice_idx : int
        Index of the slice to show in the D dimension.
    title : str
        Title for the plot.

    Returns:
    --------
    matplotlib.figure.Figure
        The Matplotlib figure for display.
    """
    slice_2d = image_3d[:, :, slice_idx]
    fig, ax = plt.subplots(figsize=(5, 5))  
    im = ax.imshow(slice_2d, cmap="gray", interpolation="none")
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")  
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def load_nifti(file_path):
    """
    Loads a NIfTI file from disk and returns its 3D data array.

    Parameters:
    -----------
    file_path : str
        Path to the NIfTI file.

    Returns:
    --------
    np.ndarray
        The data array, typically shape [D, H, W] or similar.
    """
    return nib.load(file_path).get_fdata()

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


def find_available_outputs(output_dir="./output_segmentations"):
    """
    Returns a dictionary mapping each patient_id to the paths of:
      - 'seg': segmentation file
      - 'uncertainty_NCR': path to the NCR uncertainty map
      - 'uncertainty_ED':  path to the ED uncertainty map
      - 'uncertainty_ET':  path to the ET uncertainty map
    """
    seg_files = glob.glob(os.path.join(output_dir, "hybrid_segmentation_*.nii.gz"))
    ncr_files = glob.glob(os.path.join(output_dir, "uncertainty_NCR_*.nii.gz"))
    ed_files  = glob.glob(os.path.join(output_dir, "uncertainty_ED_*.nii.gz"))
    et_files  = glob.glob(os.path.join(output_dir, "uncertainty_ET_*.nii.gz"))
    glob_files  = glob.glob(os.path.join(output_dir, "uncertainty_global_*.nii.gz"))
    softmax_files = glob.glob(os.path.join(output_dir, "hybrid_softmax_*.nii.gz"))
   
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

    return outputs

def interactive_segmentation_figure(brain_slice, seg_slice):
    fig = go.Figure()
    # Background brain image
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray',
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Segmentation overlay
    hover_text = np.where(seg_slice > 0, np.char.add("Tumor tissue: ", seg_slice.astype(str)), "No Tumor")
    fig.add_trace(go.Heatmap(
        z=seg_slice,
        colorscale=[[0, 'rgba(0,0,0,0)'], [0.33, 'red'], [0.66, 'green'], [1.0, 'blue']],
        opacity=0.5,
        text=hover_text,
        hoverinfo='text',
        showscale=False
    ))

    # Match the width you use in st.image(..., width=800)
    # You can also set the height to match the aspect ratio of your slice if needed.
    fig.update_layout(
        width=800,
        height=800,  # or compute a height that keeps the same aspect ratio as your slice
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig.update_xaxes(
        showgrid=False,      # No background grid
        zeroline=False,      # No x=0 line
        visible=False        # Hide axis ticks and labels
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )

    fig.update_yaxes(autorange='reversed')

    return fig

def interactive_single_tissue_figure(brain_slice, seg_slice, tissue, seg_color, alpha=0.3):
    """
    Creates a Plotly figure overlay for a single tissue segmentation.
    The overlay shows only the specified tissue (label) in the chosen color.
    """
    fig = go.Figure()
    # Background brain image
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray',
        showscale=False,
        hoverinfo='skip'
    ))
    # Create a binary mask for the tissue
    mask = (seg_slice == tissue).astype(float)
    hover_text = np.where(mask==1, "Tumor tissue", "No Tumor")
    
    # Convert RGB tuple to an rgba string (full opacity in the color definition)
    r, g, b = seg_color
    color_str = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
    
    fig.add_trace(go.Heatmap(
        z=mask,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, color_str]],
        opacity=alpha,
        text=hover_text,
        hoverinfo='text',
        showscale=False
    ))
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
        autorange='reversed'
    )
    return fig

def interactive_uncertainty_figure(brain_slice, unc_slice, alpha=0.5):
    """
    Creates a Plotly figure for uncertainty overlay.
    The background is the brain slice, and the uncertainty map is overlaid
    using the 'Jet' colorscale. Hovering over the uncertainty map shows its value.
    """
    fig = go.Figure()
    # Background brain image
    fig.add_trace(go.Heatmap(
        z=brain_slice,
        colorscale='gray',
        showscale=False,
        hoverinfo='skip'
    ))
    # Uncertainty overlay with hover text showing uncertainty value
    fig.add_trace(go.Heatmap(
        z=unc_slice,
        colorscale='Jet',
        opacity=alpha,
        hovertemplate='Uncertainty: %{z:.3f}<extra></extra>',
        showscale=False
    ))
    fig.update_layout(
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
        autorange='reversed'
    )
    return fig

# --------------------------------------------------------------------------------
# Additional functions for tumor counting or slice range
# --------------------------------------------------------------------------------
from scipy.ndimage import (
    label,
    generate_binary_structure,
    binary_dilation,
    center_of_mass,
)
from scipy.spatial.distance import pdist, squareform

def compute_simple_volumes(seg_data, dx, dy, dz):
    """
    Compute the total volumes (in cm³) for each BraTS-like label:
      1 -> Necrotic Core (NCR)
      2 -> Edema (ED)
      3 -> Enhancing Tumor (ET)

    Parameters
    ----------
    seg_data : np.ndarray
        A 3D array of integer labels, typically {0,1,2,3}.
    dx, dy, dz : float
        The voxel spacing (in mm) along each dimension. 
        e.g., from seg_nii.header.get_zooms().

    Returns
    -------
    (float, float, float)
        The total volumes for NCR, ED, ET in cubic centimeters.
    """
    # Each voxel volume in mm³
    voxel_vol_mm3 = dx * dy * dz

    # Convert to cm³ by dividing by 1000
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    # Count how many voxels belong to each label
    ncr_voxels = np.sum(seg_data == 1)
    ed_voxels  = np.sum(seg_data == 2)
    et_voxels  = np.sum(seg_data == 3)

    # Multiply counts by voxel volume
    ncr_volume = ncr_voxels * voxel_vol_cm3
    ed_volume  = ed_voxels  * voxel_vol_cm3
    et_volume  = et_voxels  * voxel_vol_cm3

    return ncr_volume, ed_volume, et_volume

def count_tumors_with_volume(mask_3d, voxel_vol_mm3, min_voxel_size=6, min_distance=40):
    """
    Similar to your count_tumors, but uses voxel_vol_mm3 to get real cm³ volumes.
    """
    from scipy.ndimage import label, generate_binary_structure, binary_dilation, center_of_mass
    from scipy.spatial.distance import pdist, squareform

    struct = generate_binary_structure(3, 3)
    # dilated_mask = binary_dilation(mask_3d, structure=struct, iterations=1)
    dilated_mask = mask_3d  
    labeled_mask, num_features = label(dilated_mask, structure=struct)

    # Filter out small connected components
    large_regions = []
    for i in range(1, num_features + 1):
        voxel_count = (labeled_mask == i).sum()
        if voxel_count > min_voxel_size:
            large_regions.append(i)

    if not large_regions:
        return 0, []

    if len(large_regions) == 1:
        # Only one region
        voxel_count = (labeled_mask == large_regions[0]).sum()
        volume_cm3 = (voxel_count * voxel_vol_mm3) / 1000.0
        return 1, [volume_cm3]

    # More than one region -> check distances
    centroids = [center_of_mass(mask_3d, labeled_mask, r) for r in large_regions]
    distances = squareform(pdist(centroids))

    separated_tumors = []
    for i, region_id in enumerate(large_regions):
        # If it's not "too close" to any other region, we keep it separate
        close = any(distances[i, j] < min_distance for j in range(len(large_regions)) if i != j)
        if not close:
            separated_tumors.append(region_id)

    volumes_cm3 = []
    for region_id in separated_tumors:
        voxel_count = (labeled_mask == region_id).sum()
        vol_cm3 = (voxel_count * voxel_vol_mm3) / 1000.0
        volumes_cm3.append(vol_cm3)

    return len(separated_tumors), volumes_cm3

def calculate_valid_slices(masks):
    """
    Finds the global minimum and maximum slice indices that contain any non-zero
    data among a list of 3D volumes. It was used previously for a global slider,
    but we no longer use a single global slider per design.
    Kept here if needed for some other feature.

    Parameters:
    -----------
    masks : list of np.ndarray
        Each is a 3D volume. Typically binary or integer masks.

    Returns:
    --------
    (int, int)
        The minimum and maximum slice indices that contain non-zero data across
        all volumes. If none are non-zero, returns (0,0).
    """
    valid_slices = set()
    for mask in masks:
        if mask is not None:
            non_empty_slices = set((mask != 0).any(axis=(1, 2)).nonzero()[0])
            valid_slices.update(non_empty_slices)
    if valid_slices:
        return min(valid_slices), max(valid_slices)
    return 0, 0


def create_thresholded_nifti(unc_data, threshold, original_nii):
    """
    Create two NIfTI images from the uncertainty map:
      - One for voxels with uncertainty >= threshold.
      - One for voxels with uncertainty < threshold.
    The original NIfTI is used for affine and header information.
    """
    # Preserve uncertainty values above threshold, set others to zero
    data_above = np.where(unc_data >= threshold, unc_data, 0).astype(np.float32)
    # Preserve uncertainty values below threshold, set others to zero
    data_below = np.where(unc_data < threshold, unc_data, 0).astype(np.float32)
    
    nii_above = nib.Nifti1Image(data_above, affine=original_nii.affine, header=original_nii.header)
    nii_below = nib.Nifti1Image(data_below, affine=original_nii.affine, header=original_nii.header)
    return nii_above, nii_below

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
# --------------------------------------------------------------------------------
# SimpleITK-based registration and intensity rescaling (optional)
# --------------------------------------------------------------------------------
def rescale_intensity_sitk(sitk_image):
    """
    Rescales the image intensity to [0,1] using SimpleITK.
    """
    return sitk.RescaleIntensity(sitk_image, outputMinimum=0.0, outputMaximum=1.0)

def skull_strip_with_hd_bet(input_path, output_path=None):
    """
    Run HD-BET skull stripping on a single NIfTI file.
    """
    if output_path is None:
        base = os.path.basename(input_path).replace(".nii", "").replace(".gz", "")
        output_path = os.path.join(os.getcwd(), f"{base}_skullstripped.nii.gz")

    try:
        subprocess.run([
            "hd-bet",
            "-i", input_path,
            "-o", output_path,
            "-device", "cuda"  # Change to "cpu" if needed
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[HD-BET ERROR] {e}")
        return input_path  # fallback: return original file path

    return output_path

def realign_images_to_reference(nifti_paths, progress_callback=None):
    """
    Skull-strips and realigns all images in 'nifti_paths' to the first one (as reference).
    Optionally updates a Streamlit progress bar.
    """
    if not nifti_paths:
        return []

    stripped_paths = []
    total_steps = len(nifti_paths) * 2  # Skull stripping + registration
    current_step = 0

    for p in nifti_paths:
        stripped = skull_strip_with_hd_bet(p)
        stripped_paths.append(stripped)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    reference_path = stripped_paths[0]
    ref_img_sitk = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    ref_img_sitk = rescale_intensity_sitk(ref_img_sitk)

    output_paths = []
    ref_output_path = os.path.join(os.getcwd(), f"preproc_{os.path.basename(reference_path)}")
    sitk.WriteImage(ref_img_sitk, ref_output_path)
    output_paths.append(ref_output_path)

    for path in stripped_paths[1:]:
        mov_img_sitk = sitk.ReadImage(path, sitk.sitkFloat32)
        mov_img_sitk = rescale_intensity_sitk(mov_img_sitk)

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

        out_path = os.path.join(os.getcwd(), f"preproc_{os.path.basename(path)}")
        sitk.WriteImage(aligned_img, out_path)
        output_paths.append(out_path)

        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    return output_paths

# --------------------------------------------------------------------------------
# Create Streamlit tabs
# --------------------------------------------------------------------------------
hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("🧠 Brain Tumor Segmentation")

tab1, results = st.tabs(["Data Upload & Preview", "Results"])


# --------------------------------------------------------------------------------
# TAB 1: Uploading up to 4 files, Preprocessing (optional), and Running Segmentation
# --------------------------------------------------------------------------------

with tab1:
    st.header("Welcome to the Brain Tumor Segmentation App! 👋")
    st.markdown("""
    **What does this app do?**  
    - This application lets you upload up to 4 brain MRI files in NIfTI or DICOM format.
    - You can optionally preprocess these files (intensity normalization and registration).
    - Then you can run an ensemble model to segment possible tumor regions.

    **How to use it:**  
    1. Upload your MRI files on the sidebar (up to 4).
    2. (Optional) Click **Preprocess** to run intensity rescaling and registration.
    3. Click **Run Segmentation** to generate the tumor segmentation.
    4. Switch to the **Results** tab to visualize the output.

    **Supported Modalities (BraTS style):** Flair, T1ce, T1, T2.  
    If fewer than 4 files are provided, the app duplicates some files to ensure it can still run.
    """)  

    st.sidebar.header("File Upload")

    uploaded_files = st.sidebar.file_uploader(
        "Upload up to 4 files (NIfTI or DICOM):",
        type=["nii", "nii.gz", "dcm"],
        accept_multiple_files=True,
    )

    preproc_clicked = st.sidebar.button("Preprocess", key="btn_preprocesar_sidebar")
    run_seg_clicked = st.sidebar.button("Run Tumor Segmentation", key="btn_run_segmentation")
    st.sidebar.markdown("---")

    gt_file = st.sidebar.file_uploader("Upload Ground Truth (optional)", type=["nii", "nii.gz"])
    if gt_file is not None:
        tmp_gt_path = os.path.join(os.getcwd(), f"uploaded_{gt_file.name}")
        gt_file.seek(0)
        with open(tmp_gt_path, "wb") as out:
            out.write(gt_file.read())
        st.session_state["gt_path"] = tmp_gt_path
    else:
        st.session_state["gt_path"] = None

    # --- 🧠 Session state setup ---
    if "preproc_paths" not in st.session_state:
        st.session_state["preproc_paths"] = None
    if "last_uploaded_files" not in st.session_state:
        st.session_state["last_uploaded_files"] = []

    if uploaded_files:
        uploaded_filenames = [f.name for f in uploaded_files]
        if uploaded_filenames != st.session_state["last_uploaded_files"]:
            st.session_state["preproc_paths"] = None
            st.session_state["last_uploaded_files"] = uploaded_filenames

        # --- 🧠 Dedupe and convert ---
        unique_file_names = set()
        unique_files = []
        for f in uploaded_files:
            if f.name not in unique_file_names:
                unique_file_names.add(f.name)
                unique_files.append(f)

        st.write("### File Status")
        nifti_files = []
        for file in unique_files:
            file.seek(0)
            file_format = check_file_format(file)
            if file_format == "nifti":
                st.success(f"NIfTI file: {file.name}")
                nifti_files.append(file)
            elif file_format == "dicom":
                st.warning(f"DICOM file detected: {file.name}. Converting to NIfTI...")
                file.seek(0)
                converted_path = convert_dicom_to_nifti([file], f"converted_{file.name}.nii")
                if converted_path:
                    st.info(f"Converted to NIfTI: {converted_path}")
                    nifti_files.append(converted_path)
            else:
                st.error(f"Unrecognized format: {file.name}")

        if len(nifti_files) < 4:
            st.warning("Fewer than 4 files detected. Duplicating some files to proceed.")
            while len(nifti_files) < 4 and len(nifti_files) > 0:
                nifti_files.append(nifti_files[-1])
        elif len(nifti_files) > 4:
            st.warning("More than 4 files uploaded. Only the first 4 will be used.")
            nifti_files = nifti_files[:4]

        actual_nifti_paths = []
        for item in nifti_files:
            if isinstance(item, str):
                actual_nifti_paths.append(item)
            else:
                item.seek(0)
                tmp_path = os.path.join(os.getcwd(), item.name)
                with open(tmp_path, "wb") as out:
                    out.write(item.read())
                actual_nifti_paths.append(tmp_path)

        actual_nifti_paths = reorder_modalities(actual_nifti_paths)

        # --- Run preprocessing ---
        if preproc_clicked and len(actual_nifti_paths) > 0:
            st.info("Starting preprocessing (skull stripping + registration)...")
            preproc_bar = st.progress(0, text="Preprocessing...")

            def update_progress(pct):
                preproc_bar.progress(int(pct * 100), text=f"Preprocessing... {int(pct * 100)}%")

            result = realign_images_to_reference(actual_nifti_paths, progress_callback=update_progress)
            st.session_state["preproc_paths"] = result
            preproc_bar.progress(100, text="✅ Preprocessing complete!")
            st.success("Preprocessing completed.")

        preproc_paths = st.session_state.get("preproc_paths")
        if preproc_paths is None:
            # Use the raw uploaded NIfTI paths
            preproc_paths = actual_nifti_paths

        # --- Run segmentation ---
        st.markdown("---")
        if run_seg_clicked:
            st.subheader("Running Segmentation")
            if not preproc_paths or not os.path.exists(preproc_paths[0]):
                st.error("Preprocessed (or raw) file not found.")
            else:
                st.info("Loading the model...")
                models_dict = load_ensemble_models()
                st.info("Loading data...")
                test_loader = create_test_loader(preproc_paths)
                patient_id = fem.extract_patient_id(preproc_paths[0]) or None
                progress_bar = st.progress(0, text="Running segmentation...")
                fem.ensemble_segmentation(
                    test_loader, 
                    models_dict, 
                    composite_score_weights={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}, 
                    n_iterations=10, 
                    progress_bar=progress_bar  
                )
                st.success("Segmentation complete! Check the 'Results' tab for output.")

        # --- Image previews ---
        st.write("## Preview of Uploaded Files")

        st.subheader("🧠 Original Scans")
        example_img = nib.load(actual_nifti_paths[0])
        max_slices_orig = example_img.shape[-1]
        slice_idx_orig = st.slider("Slice Index (Original)", 0, max_slices_orig - 1, max_slices_orig // 2, key="orig_slider")
        cols = st.columns(len(actual_nifti_paths))
        for i, path in enumerate(actual_nifti_paths):
            with cols[i]:
                slice_img = load_slice_lazy(path, slice_idx_orig)
                st.write(f"**{os.path.basename(path)}**")
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(slice_img, cmap="gray")
                ax.axis("off")
                st.pyplot(fig)

        if preproc_clicked and preproc_paths and all(os.path.exists(p) for p in preproc_paths):
            if (preproc_paths and preproc_paths != actual_nifti_paths and all(os.path.exists(p) for p in preproc_paths)):
                st.subheader("🧼 Preprocessed Scans")
                example_img_pre = nib.load(preproc_paths[0])
                max_slices_pre = example_img_pre.shape[-1]
                slice_idx_pre = st.slider("Slice Index (Preprocessed)", 0, max_slices_pre - 1, max_slices_pre // 2, key="preproc_slider")
                cols = st.columns(len(preproc_paths))
                for i, path in enumerate(preproc_paths):
                    with cols[i]:
                        slice_img = load_slice_lazy(path, slice_idx_pre)
                        st.write(f"**{os.path.basename(path)}**")
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(slice_img, cmap="gray")
                        ax.axis("off")
                        st.pyplot(fig)

# --------------------------------------------------------------------------------
# TAB 2: RESULTS
# --------------------------------------------------------------------------------
with results:
    st.header("Results & Visualization")
    all_outputs = find_available_outputs("./output_segmentations")
    if not all_outputs:
        st.warning("No segmentations found in ./output_segmentations. Please run a segmentation first.")
    else:
        # Select patient ID and load original file if provided
        patient_ids = sorted(all_outputs.keys())
        chosen_pid = st.selectbox("Select a patient ID:", patient_ids)
        original_file = st.file_uploader("Upload the original scan for overlay (NIfTI):", type=["nii","nii.gz"])
        original_path = None
        brain_data = None
        if original_file is not None:
            original_file.seek(0)
            tmp_orig = os.path.join(os.getcwd(), original_file.name)
            with open(tmp_orig, "wb") as out:
                out.write(original_file.read())
            original_path = tmp_orig
            brain_data = nib.load(original_path).get_fdata()

        # Retrieve segmentation and uncertainty file paths
        seg_path = all_outputs[chosen_pid].get("seg", None)
        ncr_path = all_outputs[chosen_pid].get("uncertainty_NCR", None)
        ed_path  = all_outputs[chosen_pid].get("uncertainty_ED", None)
        et_path  = all_outputs[chosen_pid].get("uncertainty_ET", None)
        global_unc_path = all_outputs[chosen_pid].get("uncertainty_global", None)

        # Checkboxes to toggle segmentation / uncertainty
        show_seg = st.checkbox("Show Segmentation", value=True)
        show_prob = st.checkbox("Show Probability Map", value=False)
        show_unc = st.checkbox("Show Uncertainty", value=False)

        col_seg_options, col_prob_options, col_unc_options = st.columns(3)

        # -----------------------------------------------------------
        # SEGMENTATION OPTIONS
        # -----------------------------------------------------------
        selected_tissues = []
        if show_seg:
            with col_seg_options:
                st.markdown("### Select Segmentation Tissues to Display")
                show_ncr = st.checkbox("Necrotic Core (Label 1)", value=False)
                show_ed  = st.checkbox("Edema (Label 2)", value=False)
                show_et  = st.checkbox("Enhancing Tumor (Label 3)", value=False)
                
                if show_ncr:
                    selected_tissues.append((1, (1, 0, 0)))  # Label 1, red
                if show_ed:
                    selected_tissues.append((2, (0, 1, 0)))  # Label 2, green
                if show_et:
                    selected_tissues.append((3, (0, 0, 1)))  # Label 3, blue
            
            if brain_data is None:
                st.warning("Select a brain scan to show the segmentation")
        
        # -----------------------------------------------------------
        # PROBABILITY OPTIONS
        # -----------------------------------------------------------
        selected_probs = []
        prob_path = all_outputs[chosen_pid].get("softmax", None)
        if show_prob and prob_path:
            with col_prob_options:
                st.markdown("### Select Probabilities to Display")

                show_prob_ncr = st.checkbox("NCR Probability", value=False)
                show_prob_ed  = st.checkbox("ED Probability", value=False)
                show_prob_et  = st.checkbox("ET Probability", value=False)

                softmax_data = nib.load(prob_path).get_fdata()  # shape: [4, H, W, D]
                if show_prob_ncr:
                    selected_probs.append(("NCR", softmax_data[1]))  # label 1
                if show_prob_ed:
                    selected_probs.append(("ED",  softmax_data[2]))  # label 2
                if show_prob_et:
                    selected_probs.append(("ET",  softmax_data[3]))  # label 3
        elif show_prob and not prob_path:
            st.warning("No softmax probability file found for this patient.")

        # -----------------------------------------------------------
        # UNCERTAINTY OPTIONS
        # -----------------------------------------------------------
        selected_uncertainties = []
        if show_unc:
            with col_unc_options:
                st.markdown("#### Select Uncertainty Map(s) to Display")
                show_unc_ncr = st.checkbox("NCR Uncertainty", value=False)
                show_unc_ed  = st.checkbox("ED Uncertainty", value=False)
                show_unc_et  = st.checkbox("ET Uncertainty", value=False)
                
                if show_unc_ncr and ncr_path:
                    selected_uncertainties.append(("NCR", nib.load(ncr_path).get_fdata()))
                if show_unc_ed and ed_path:
                    selected_uncertainties.append(("ED", nib.load(ed_path).get_fdata()))
                if show_unc_et and et_path:
                    selected_uncertainties.append(("ET", nib.load(et_path).get_fdata()))
            
            if brain_data is None:
                st.warning("Select a brain scan to show uncertainty maps")

        # Warn if seg file is missing
        if show_seg and not seg_path:
            st.warning("No segmentation file found for this patient.")

        # Determine how many slices
        if brain_data is not None:
            depth = brain_data.shape[-1]
        else:
            fallback_data = None
            for possible in [seg_path, ncr_path, ed_path, et_path, global_unc_path]:
                if possible:
                    fallback_data = nib.load(possible).get_fdata()
                    break
            depth = fallback_data.shape[-1] if fallback_data is not None else 0

        if depth > 0:
            st.markdown("#### Slice Index")
            slice_idx = st.slider("Slice Index", 0, depth - 1, depth // 2, key="slice_slider")
        else:
            st.warning("No data found to slice.")
            slice_idx = 0

        # -----------------------------------------------------------
        # OPACITY SLIDERS
        # -----------------------------------------------------------
        if show_seg or show_unc:
            st.markdown("### Overlay Settings")
            col1, col2, col3 = st.columns(3)
            seg_opacity = col1.slider("Segmentation Opacity", 0.0, 1.0, 0.4, 0.01, key="seg_op_slider")
            prob_opacity = col2.slider("Probability Opacity", 0.0, 1.0, 0.4, 0.01, key="prob_op_slider")
            unc_opacity = col3.slider("Uncertainty Opacity", 0.0, 1.0, 0.4, 0.01, key="unc_op_slider")
        else:
            seg_opacity = 0.4
            prob_opacity = 0.4
            unc_opacity = 0.4

        seg_figure = None
        unc_figure = None
        threshold = None

        # -----------------------------------------------------------
        # BUILD THE FIGURES
        # -----------------------------------------------------------
        if brain_data is not None:
            brain_slice = brain_data[..., slice_idx]

            # --- SEGMENTATION FIGURE ---
            if show_seg and seg_path:
                if selected_tissues:
                    seg_data = nib.load(seg_path).get_fdata()
                    seg_slice = seg_data[..., slice_idx]
                    seg_figure = go.Figure()
                    # Add background brain slice
                    seg_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip',
                        zsmooth="best"
                    ))

                    label_names = {
                        1: "Necrotic Core",
                        2: "Edema",
                        3: "Enhancing Tumor"
                    }

                    # Sort tissues so that lower labels are drawn first
                    selected_tissues.sort(key=lambda x: x[0])
                    overlay_traces = []
                    composite_text = np.full(seg_slice.shape, "", dtype='<U50')

                    for label_val, color in selected_tissues:
                        mask = (seg_slice == label_val)
                        z_data = mask.astype(float)
                        r, g, b = color
                        rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
                        overlay_traces.append(go.Heatmap(
                            z=z_data,
                            colorscale=[[0, 'rgba(0,0,0,0)'], [1, rgba]],
                            opacity=seg_opacity,
                            hoverinfo='skip',
                            showscale=False
                        ))
                        label_str = label_names.get(label_val, f"Tissue {label_val}")
                        composite_text[mask] = label_str

                    for trace in overlay_traces:
                        seg_figure.add_trace(trace)

                    seg_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        text=composite_text,
                        hoverinfo='text',
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                        opacity=1,
                        showscale=False,
                        zsmooth="best"
                    ))

                    seg_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )

                else:
                    seg_figure = go.Figure(go.Heatmap(
                        z=brain_slice,
                        colorscale="gray",
                        showscale=False
                    ))
                    seg_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )


            # --- (OPTIONAL) GROUND TRUTH FIGURE ---
            # If a ground truth file was uploaded in Tab 1, it is stored in st.session_state["gt_path"]
            if show_seg and seg_figure and st.session_state.get("gt_path") is not None:
                gt_data = nib.load(st.session_state["gt_path"]).get_fdata()
                gt_slice = gt_data[..., slice_idx]
                gt_figure = go.Figure()
                gt_figure.add_trace(go.Heatmap(
                    z=brain_slice,
                    colorscale='gray',
                    showscale=False,
                    hoverinfo='skip',
                    zsmooth="best"
                ))
                # We assume the ground truth segmentation uses the same labels:
                label_names = {1: "Necrotic Core", 2: "Edema", 3: "Enhancing Tumor"}
                for label_val, color in [(1, (1, 0, 0)), (2, (0, 1, 0)), (3, (0, 0, 1))]:
                    mask = (gt_slice == label_val)
                    if np.any(mask):
                        z_data = mask.astype(float)
                        r, g, b = color
                        rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
                        gt_figure.add_trace(go.Heatmap(
                            z=z_data,
                            colorscale=[[0, 'rgba(0,0,0,0)'], [1, rgba]],
                            opacity=seg_opacity,
                            hoverinfo='skip',
                            showscale=False,
                            zsmooth="best"
                        ))
                # Create composite hover text for ground truth
                composite_text = np.full(gt_slice.shape, "", dtype='<U50')
                for label_val in [1, 2, 3]:
                    composite_text = np.where(gt_slice == label_val, label_names.get(label_val, f"Tissue {label_val}"), composite_text)
                gt_figure.add_trace(go.Heatmap(
                    z=brain_slice,
                    text=composite_text,
                    hoverinfo='text',
                    hovertemplate="%{text}<extra></extra>",
                    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                    opacity=1,
                    showscale=False,
                    zsmooth="best"
                ))
                gt_figure.update_layout(
                    width=800,
                    height=800,
                    margin=dict(l=0, r=0, t=20, b=80),
                    xaxis=dict(visible=False),
                    yaxis=dict(
                        visible=False,
                        autorange='reversed',
                        scaleanchor="x",  # lock aspect ratio
                        scaleratio=1
                    )
                )



            # --- PROBABILITY MAP FIGURE ---
            prob_figure = None
            if show_prob:
                if selected_probs:
                    prob_figure = go.Figure()
                    prob_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        colorscale="gray",
                        showscale=False,
                        hoverinfo="skip",
                        zsmooth="best"
                    ))
                    color_map = {
                        "NCR": (255, 0, 0),
                        "ED":  (0, 255, 0),
                        "ET":  (0, 0, 255)
                    }
                    composite_text = np.full(brain_slice.shape, "", dtype="<U200")
                    for label, volume in selected_probs:
                        prob_slice = volume[..., slice_idx]
                        r, g, b = color_map[label]
                        rgba = f"rgba({r},{g},{b},1)"
                        formatted = np.vectorize(lambda p: f"{label} Prob: {p:.3f}")(prob_slice)
                        composite_text = np.where(
                            composite_text == "",
                            formatted,
                            np.char.add(np.char.add(composite_text, "<br>"), formatted)
                        )
                        prob_figure.add_trace(go.Heatmap(
                            z=prob_slice,
                            colorscale=[[0, "rgba(0,0,0,0)"], [1, rgba]],
                            opacity=prob_opacity, 
                            hoverinfo="skip",
                            showscale=False,
                            zsmooth="best"
                        ))
                    prob_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        text=composite_text,
                        hoverinfo='text',
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                        opacity=1,
                        showscale=False,
                        zsmooth="best"
                    ))
                    prob_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )
                else:
                    prob_figure = go.Figure(go.Heatmap(
                        z=brain_slice,
                        colorscale="gray",
                        showscale=False
                    ))
                    prob_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )
                    

            # --- UNCERTAINTY FIGURE ---
            if show_unc:
                if selected_uncertainties:
                    unc_figure = go.Figure()
                    unc_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        colorscale='gray',
                        showscale=False,
                        hoverinfo='skip',
                        zsmooth="best"
                    ))
                    st.markdown("#### Uncertainty Threshold")
                    threshold_mode = st.radio(
                        "Threshold Mode:",
                        ["Below", "Above"],
                        index=0,
                        horizontal=True
                    )
                    threshold = st.slider(
                        "Only show uncertainty values BELOW or ABOVE this threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.01,
                        key="threshold_slider"
                    )
                    unc_colors = {
                        "NCR": (1, 0, 0),
                        "ED":  (0, 1, 0),
                        "ET":  (0, 0, 1),
                    }
                    overlay_traces = []
                    composite_text = np.full(brain_slice.shape, "", dtype='<U50')
                    for label, unc_data in selected_uncertainties:
                        if threshold_mode == "Below":
                            unc_thresholded = np.where(unc_data <= threshold, unc_data, 0.0)
                        else:
                            unc_thresholded = np.where(unc_data >= threshold, unc_data, 0.0)
                        unc_slice = unc_thresholded[..., slice_idx]
                        fmt = np.vectorize(lambda x: f"{x:.3f}")
                        formatted_values = fmt(unc_slice)
                        r, g, b = unc_colors[label]
                        rgba = f'rgba({int(r*255)},{int(g*255)},{int(b*255)},1)'
                        trace = go.Heatmap(
                            z=unc_slice,
                            colorscale=[[0, 'rgba(0,0,0,0)'], [1, rgba]],
                            opacity=unc_opacity,
                            hoverinfo='skip',
                            showscale=False
                        )
                        overlay_traces.append(trace)
                        label_str_array = np.full(brain_slice.shape, f"{label}: ", dtype='<U50')
                        composite_text = np.where(
                            composite_text == "",
                            np.char.add(label_str_array, formatted_values),
                            np.char.add(composite_text, np.char.add("<br>", np.char.add(label_str_array, formatted_values)))
                        )
                    for trace in overlay_traces:
                        unc_figure.add_trace(trace)
                    unc_figure.add_trace(go.Heatmap(
                        z=brain_slice,
                        text=composite_text,
                        hoverinfo='text',
                        hovertemplate="%{text}<extra></extra>",
                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                        opacity=1,
                        showscale=False,
                        zsmooth="best"
                    ))
                    unc_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )
                else:
                    unc_figure = go.Figure(go.Heatmap(
                        z=brain_slice,
                        colorscale="gray",
                        showscale=False
                    ))
                    unc_figure.update_layout(
                        width=800,
                        height=800,
                        margin=dict(l=0, r=0, t=20, b=80),
                        xaxis=dict(visible=False),
                        yaxis=dict(
                            visible=False,
                            autorange='reversed',
                            scaleanchor="x",  # lock aspect ratio
                            scaleratio=1
                        )
                    )

            # --- DISPLAY FIGURES ---
            # If ground truth exists, show the predicted segmentation and ground truth side by side.
            if show_seg and seg_figure:
                if st.session_state.get("gt_path") is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Predicted Segmentation")
                        st.plotly_chart(seg_figure, use_container_width=True)
                    with col2:
                        st.markdown("#### Ground Truth Segmentation")
                        st.plotly_chart(gt_figure, use_container_width=True)
                else:
                    st.markdown("#### Predicted Segmentation")
                    st.plotly_chart(seg_figure, use_container_width=True)

            # Display other figures (Probability, Uncertainty) in a separate row
            cols_extra = []
            if show_prob and prob_figure: 
                cols_extra.append(("Probabilitiy map", prob_figure))
            if show_unc and unc_figure: 
                cols_extra.append(("Uncertainty map", unc_figure))
            if cols_extra:
                plot_cols = st.columns(len(cols_extra))
                for i, ((label, fig), col) in enumerate(zip(cols_extra, plot_cols)):
                    with col:
                        st.markdown(f"#### {label}")
                        st.plotly_chart(fig, use_container_width=True, key=f"{label.lower()}_chart_{i}")

            # Tumor volumes, etc...
            if seg_path is not None:
                seg_nii = nib.load(seg_path)
                seg_data = seg_nii.get_fdata()
                dx, dy, dz = seg_nii.header.get_zooms()
                ncr_vol, ed_vol, et_vol = compute_simple_volumes(seg_data, dx, dy, dz)
                total_vol = ncr_vol + ed_vol + et_vol

                st.subheader("Tumor Volumes")
                col_ncr, col_ed, col_et, col_total = st.columns(4)
                with col_ncr:
                    st.metric(label="NCR Volume", value=f"{ncr_vol:.2f} cm³")
                with col_ed:
                    st.metric(label="Edema Volume", value=f"{ed_vol:.2f} cm³")
                with col_et:
                    st.metric(label="Enhancing Tumor Volume", value=f"{et_vol:.2f} cm³")
                with col_total:
                    st.metric(label="Total Tumor Volume", value=f"{total_vol:.2f} cm³")

            # Download thresholded uncertainty maps if desired...
            if show_unc and selected_uncertainties and threshold is not None:
                st.markdown("### Download Thresholded Uncertainty Maps")
                for label, unc_data in selected_uncertainties:
                    if label == "NCR":
                        unc_path = ncr_path
                    elif label == "ED":
                        unc_path = ed_path
                    elif label == "ET":
                        unc_path = et_path
                    else:
                        unc_path = None

                    if unc_path is not None:
                        unc_nii = nib.load(unc_path)
                        below_data = np.where(unc_data <= threshold, unc_data, 0.0)
                        above_data = np.where(unc_data > threshold, unc_data, 0.0)

                        nii_below = nib.Nifti1Image(below_data, affine=unc_nii.affine, header=unc_nii.header)
                        nii_above = nib.Nifti1Image(above_data, affine=unc_nii.affine, header=unc_nii.header)

                        bytes_below = nifti_to_bytes(nii_below)
                        bytes_above = nifti_to_bytes(nii_above)

                        st.download_button(
                            label=f"Download {label} Map (Below Threshold)",
                            data=bytes_below,
                            file_name=f"{label}_uncertainty_below_{threshold}_threshold.nii.gz",
                            mime="application/octet-stream"
                        )
                        st.download_button(
                            label=f"Download {label} Map (Above Threshold)",
                            data=bytes_above,
                            file_name=f"{label}_uncertainty_above_{threshold}_threshold.nii.gz",
                            mime="application/octet-stream"
                        )

            # Optional: Download segmentation figure
            if show_seg and seg_figure is not None:
                img_bytes = seg_figure.to_image(format="png")
                st.download_button(
                    label="Download Segmentation Figure",
                    data=img_bytes,
                    file_name="segmentation_figure.png",
                    mime="image/png"
                )
