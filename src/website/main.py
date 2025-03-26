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

    return outputs

def overlay_segmentation(brain_slice, seg_slice, alpha=0.3, segmentation_colors={1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1],}):
    """
    Overlays a segmentation mask onto a brain slice.
    Each tissue label is blended with its specified color.
    
    Parameters:
      - brain_slice: 2D array (grayscale) of the original scan.
      - seg_slice: 2D array of segmentation labels.
      - alpha: blending factor.
      - segmentation_colors: dict mapping label -> RGB color.
    
    Returns:
      A 3D array representing an RGB image with the overlay.
    """
    bmin, bmax = brain_slice.min(), brain_slice.max()
    brain_norm = (brain_slice - bmin) / (bmax - bmin + 1e-8)
    brain_rgb = np.stack([brain_norm]*3, axis=-1)

    overlay_rgb = brain_rgb.copy()

    # Loop through each label in the segmentation and apply its color
    for label, color in segmentation_colors.items():
        mask = seg_slice == label
        # Blend each pixel that belongs to this label
        overlay_rgb[mask] = (1 - alpha) * overlay_rgb[mask] + alpha * np.array(color)
    
    return overlay_rgb

def overlay_single_tissue(brain_slice, seg_slice, tissue, alpha=0.3, seg_color=(1,0,0)):
    """
    Overlays a segmentation mask (seg_slice) on top of a grayscale brain_slice.
    seg_slice is expected to be label-based or binary (0=bg, >0=fg).
    Returns an RGB image (H,W,3).
    """
    bmin, bmax = brain_slice.min(), brain_slice.max()
    brain_norm = (brain_slice - bmin) / (bmax - bmin + 1e-8)
    brain_rgb = np.stack([brain_norm]*3, axis=-1)

    overlay_rgb = brain_rgb.copy()
    mask = seg_slice == tissue
    overlay_rgb[mask] = (1 - alpha)*overlay_rgb[mask] + alpha*np.array(seg_color)
    return overlay_rgb

def overlay_uncertainty(brain_slice, unc_slice, alpha=0.5, cmap=plt.cm.jet):
    """
    Overlays an uncertainty map (unc_slice) on top of a grayscale brain_slice.
    unc_slice should be in [0,1] or you can rescale it first.
    Returns an RGB image (H,W,3).
    """
    bmin, bmax = brain_slice.min(), brain_slice.max()
    brain_norm = (brain_slice - bmin) / (bmax - bmin + 1e-8)
    brain_rgb = np.stack([brain_norm]*3, axis=-1)

    # Normalize unc_slice
    umin, umax = unc_slice.min(), unc_slice.max()
    if umax - umin < 1e-8:
        unc_norm = np.zeros_like(unc_slice)
    else:
        unc_norm = (unc_slice - umin)/(umax - umin)
    unc_colored = cmap(unc_norm)[..., :3]  # (H,W,3)

    # Alpha blend
    overlay_rgb = (1 - alpha)*brain_rgb + alpha*unc_colored
    return overlay_rgb

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

    # Match the width you use in st.image(..., width=400)
    # You can also set the height to match the aspect ratio of your slice if needed.
    fig.update_layout(
        width=400,
        height=400,  # or compute a height that keeps the same aspect ratio as your slice
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
        width=400,
        height=400,
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
        width=400,
        height=400,
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


def count_tumors(mask, min_voxel_size=6, min_distance=40):
    """
    Counts tumors in a 3D binary mask. Each connected component above a size threshold
    is considered. If multiple tumors are close (below min_distance in centroid space),
    they are considered the same.

    Parameters:
    -----------
    mask : np.ndarray
        3D binary array (e.g., [D,H,W]) where non-zero indicates tumor.
    min_voxel_size : int
        Minimum connected component size in voxels to be considered valid.
    min_distance : float
        Minimum distance (in voxel units) between centroids to consider them separate.

    Returns:
    --------
    (int, list)
        The number of separate tumors, and a list of volumes (in cm^3) of each.
    """
    struct = generate_binary_structure(3, 3)
    dilated_mask = binary_dilation(mask, structure=struct, iterations=1)
    labeled_mask, num_features = label(dilated_mask, structure=struct)

    # Filter out small connected components
    large_regions = [
        i
        for i in range(1, num_features + 1)
        if (labeled_mask == i).sum() > min_voxel_size
    ]
    if len(large_regions) == 0:
        return 0, []

    if len(large_regions) == 1:
        volume_voxels = (labeled_mask == large_regions[0]).sum()
        volume_cm3 = volume_voxels / 1000
        return 1, [volume_cm3]

    centroids = [
        center_of_mass(mask, labeled_mask, region_id) for region_id in large_regions
    ]
    distances = squareform(pdist(centroids))
    separated_tumors = []
    for i, region_id in enumerate(large_regions):
        close = any(
            distances[i, j] < min_distance for j in range(len(large_regions)) if i != j
        )
        if not close:
            separated_tumors.append(region_id)

    tumor_volumes = []
    for region_id in separated_tumors:
        volume_voxels = (labeled_mask == region_id).sum()
        volume_cm3 = volume_voxels / 1000
        tumor_volumes.append(volume_cm3)

    return len(separated_tumors), tumor_volumes


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
                st.info("Loading ensemble models...")
                models_dict = load_ensemble_models()
                st.info("Creating test loader...")
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
        seg_path = all_outputs[chosen_pid].get("seg", None)
        ncr_path = all_outputs[chosen_pid].get("uncertainty_NCR", None)
        ed_path  = all_outputs[chosen_pid].get("uncertainty_ED", None)
        et_path  = all_outputs[chosen_pid].get("uncertainty_ET", None)
        global_unc_path = all_outputs[chosen_pid].get("uncertainty_global", None)
        show_seg = st.checkbox("Show Segmentation", value=True)
        show_unc = st.checkbox("Show Uncertainty", value=False)
        seg_type = st.selectbox("Segmentation Tissue", ["Necrotic Core", "Edema", "Enhancing Tumor", "All"])
        unc_type = st.selectbox("Uncertainty Tissue", ["Necrotic Core", "Edema", "Enhancing Tumor", "All"])
        if show_seg and not seg_path:
            st.warning("No segmentation file found for this patient.")
        if show_unc:
            if unc_type == "Necrotic Core" and not ncr_path:
                st.warning("No Necrotic Core uncertainty found for this patient.")
            elif unc_type == "Edema" and not ed_path:
                st.warning("No Edema uncertainty found for this patient.")
            elif unc_type == "Enhancing Tumor" and not et_path:
                st.warning("No Enhancing Tumor uncertainty found for this patient.")
            elif unc_type == "All" and not global_unc_path:
                st.warning("No global uncertainty found for this patient.")
        if brain_data is None:
            st.info("Upload an original scan to see overlays. Otherwise, we'll just show the raw segmentation/uncertainty.")
        depth = 0
        if brain_data is not None:
            depth = brain_data.shape[-1]
        else:
            fallback_data = None
            if seg_path:
                fallback_data = nib.load(seg_path).get_fdata()
            elif ncr_path:
                fallback_data = nib.load(ncr_path).get_fdata()
            elif ed_path:
                fallback_data = nib.load(ed_path).get_fdata()
            elif et_path:
                fallback_data = nib.load(et_path).get_fdata()
            elif global_unc_path:
                fallback_data = nib.load(global_unc_path).get_fdata()
            if fallback_data is not None:
                depth = fallback_data.shape[-1]
        if depth > 0:
            slice_idx = st.slider("Slice Index", 0, depth-1, depth//2)
        else:
            st.warning("No data found to slice.")
            slice_idx = 0
        
        seg_figure = None
        unc_figure = None
        if brain_data is not None:
            brain_slice = brain_data[..., slice_idx]
            if show_seg and seg_path:
                seg_data = nib.load(seg_path).get_fdata()
                seg_slice = seg_data[..., slice_idx]
                if seg_type == "Necrotic Core" and ncr_path:
                    seg_figure = interactive_single_tissue_figure(brain_slice, seg_slice, tissue=1, seg_color=(1,0,0), alpha=0.3)
                elif seg_type == "Edema" and ed_path:
                    seg_figure = interactive_single_tissue_figure(brain_slice, seg_slice, tissue=2, seg_color=(0,1,0), alpha=0.3)
                elif seg_type == "Enhancing Tumor" and et_path:
                    seg_figure = interactive_single_tissue_figure(brain_slice, seg_slice, tissue=3, seg_color=(0,0,1), alpha=0.3)
                elif seg_type == "All":
                    seg_figure = interactive_segmentation_figure(brain_slice, seg_slice)
                else:
                    seg_figure = go.Figure(go.Heatmap(z=brain_slice, colorscale='gray', showscale=False))
                    seg_figure.update_layout(width=400, height=400, margin=dict(l=0,r=0,t=0,b=0))
                seg_figure.update_layout(
                    width=400,
                    height=450,  # slightly taller to allow room for caption
                    margin=dict(l=0, r=0, t=20, b=80),  # extra bottom margin
                    annotations=[
                        go.layout.Annotation(
                            text=f"Segmentation for {chosen_pid}, Slice {slice_idx}",
                            x=0.5,            # center horizontally
                            y=-0.15,          # a negative value to push it below the plot
                            xref="paper",
                            yref="paper",
                            xanchor="center",
                            yanchor="top",
                            showarrow=False,
                            font=dict(size=16)
                        )
                    ]
                )

            if show_unc:
                unc_data = None
                if unc_type == "Necrotic Core" and ncr_path:
                    unc_data = nib.load(ncr_path).get_fdata()
                elif unc_type == "Edema" and ed_path:
                    unc_data = nib.load(ed_path).get_fdata()
                elif unc_type == "Enhancing Tumor" and et_path:
                    unc_data = nib.load(et_path).get_fdata()
                elif unc_type == "All" and global_unc_path:
                    unc_data = nib.load(global_unc_path).get_fdata()
                if unc_data is not None:
                    unc_slice = unc_data[..., slice_idx]
                    unc_figure = interactive_uncertainty_figure(brain_slice, unc_slice, alpha=0.5)
                    unc_figure.update_layout(
                        width=400,
                        height=450,  # slightly taller to allow room for caption
                        margin=dict(l=0, r=0, t=20, b=80),  # extra bottom margin
                        annotations=[
                            go.layout.Annotation(
                                text=f"Uncertainty for {chosen_pid}, Slice {slice_idx}",
                                x=0.5,            # center horizontally
                                y=-0.15,          # a negative value to push it below the plot
                                xref="paper",
                                yref="paper",
                                xanchor="center",
                                yanchor="top",
                                showarrow=False,
                                font=dict(size=16)
                            )
                        ]
                    )
                    
            if seg_figure is not None and unc_figure is not None:
                col_seg, col_unc = st.columns(2)
                with col_seg:
                    st.plotly_chart(seg_figure, use_container_width=False)
                with col_unc:
                    st.plotly_chart(unc_figure, use_container_width=False)
            elif seg_figure is not None:
                col1, col2, col3 = st.columns([1,6,1])
                with col2:
                    st.plotly_chart(seg_figure, use_container_width=False)
            elif unc_figure is not None:
                col1, col2, col3 = st.columns([1,6,1])
                with col2:
                    st.plotly_chart(unc_figure, use_container_width=False)
        else:
            if show_seg and seg_path:
                seg_data = nib.load(seg_path).get_fdata()
                seg_slice = seg_data[..., slice_idx]
                col1, col2, col3 = st.columns([1,6,1])
                with col2:
                    st.image(seg_slice, caption=f"Segmentation for {chosen_pid}, Slice {slice_idx}",
                            width=400, clamp=True)
            if show_unc:
                unc_data = None
                if unc_type == "Necrotic Core" and ncr_path:
                    unc_data = nib.load(ncr_path).get_fdata()
                elif unc_type == "Edema" and ed_path:
                    unc_data = nib.load(ed_path).get_fdata()
                elif unc_type == "Enhancing Tumor" and et_path:
                    unc_data = nib.load(et_path).get_fdata()
                elif unc_type == "All" and global_unc_path:
                    unc_data = nib.load(global_unc_path).get_fdata()
                if unc_data is not None:
                    unc_slice = unc_data[..., slice_idx]
                    col1, col2, col3 = st.columns([1,6,1])
                    with col2:
                        st.image(unc_slice, caption=f"Uncertainty for {chosen_pid}, Slice {slice_idx}",
                            width=400, clamp=True)

