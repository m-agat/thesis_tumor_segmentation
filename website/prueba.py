import streamlit as st
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import os

# Optional: SimpleITK for registration and intensity rescaling
import SimpleITK as sitk

print(os.getcwd())

# --------------------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tumor Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------

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
        if file.name.endswith('.nii') or file.name.endswith('.nii.gz'):
            return "nifti"
        elif file.name.endswith('.dcm') or pydicom.dcmread(file, stop_before_pixels=True):
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
    fig, ax = plt.subplots(figsize=(6, 6))
    # Show the selected slice
    im = ax.imshow(image_3d[slice_idx, :, :], cmap="gray", interpolation="none")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Intensity")
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


# --------------------------------------------------------------------------------
# Additional functions for tumor counting or slice range
# --------------------------------------------------------------------------------
from scipy.ndimage import (
    label, 
    generate_binary_structure, 
    binary_dilation, 
    center_of_mass
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
        i for i in range(1, num_features + 1)
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


def realign_images_to_reference(nifti_paths):
    """
    Realigns all images in 'nifti_paths' to the first one (considered the reference). 
    Uses a basic Euler3D transform initializer with MeanSquares and linear interpolation.

    Parameters:
    -----------
    nifti_paths : list of str
        Paths to the NIfTI files.

    Returns:
    --------
    list of str
        List of paths to the newly resampled and rescaled images.
    """
    if not nifti_paths:
        return []

    # The first file is the reference
    reference_path = nifti_paths[0]
    ref_img_sitk = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    ref_img_sitk = rescale_intensity_sitk(ref_img_sitk)

    output_paths = []
    ref_output_path = os.path.join(os.getcwd(), f"preproc_{os.path.basename(reference_path)}")
    sitk.WriteImage(ref_img_sitk, ref_output_path)
    output_paths.append(ref_output_path)

    for path in nifti_paths[1:]:
        mov_img_sitk = sitk.ReadImage(path, sitk.sitkFloat32)
        mov_img_sitk = rescale_intensity_sitk(mov_img_sitk)

        # Basic geometry-based initial transform
        initial_transform = sitk.CenteredTransformInitializer(
            ref_img_sitk,
            mov_img_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, numberOfIterations=40
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        final_transform = registration_method.Execute(
            sitk.Cast(ref_img_sitk, sitk.sitkFloat32),
            sitk.Cast(mov_img_sitk, sitk.sitkFloat32)
        )

        # Resample the moving image into the reference space
        aligned_img = sitk.Resample(
            mov_img_sitk,
            ref_img_sitk,
            final_transform,
            sitk.sitkLinear,
            0.0,
            mov_img_sitk.GetPixelID()
        )

        out_path = os.path.join(os.getcwd(), f"preproc_{os.path.basename(path)}")
        sitk.WriteImage(aligned_img, out_path)
        output_paths.append(out_path)

    return output_paths

# --------------------------------------------------------------------------------
# Create Streamlit tabs
# --------------------------------------------------------------------------------
tab1, results = st.tabs(["Before Model", "Results"])


# --------------------------------------------------------------------------------
# TAB 1: Uploading up to 4 files, Preprocessing, optional GT
# --------------------------------------------------------------------------------
with tab1:
    st.sidebar.header("File Upload")

    # 1) Upload up to 4 files
    uploaded_files = st.sidebar.file_uploader(
        "Upload up to 4 files (NIfTI or DICOM):",
        type=["nii", "nii.gz", "dcm"],
        accept_multiple_files=True
    )

    # 2) "Preprocess" button
    preproc_clicked = st.sidebar.button("Preprocess", key="btn_preprocesar_sidebar")

    # 3) Horizontal separator
    st.sidebar.markdown("---")

    # 4) Optional Ground Truth upload
    gt_file = st.sidebar.file_uploader(
        "Upload Ground Truth (optional)",
        type=["nii", "nii.gz"]
    )

    # Save the GT path into session_state if provided
    if gt_file is not None:
        tmp_gt_path = os.path.join(os.getcwd(), f"uploaded_{gt_file.name}")
        gt_file.seek(0)
        with open(tmp_gt_path, "wb") as out:
            out.write(gt_file.read())
        st.session_state["gt_path"] = tmp_gt_path
    else:
        st.session_state["gt_path"] = None

    if uploaded_files:
        # Deduplicate files by name
        unique_file_names = set()
        unique_files = []
        for f in uploaded_files:
            if f.name not in unique_file_names:
                unique_file_names.add(f.name)
                unique_files.append(f)

        st.write("### File Status")
        nifti_files = []

        # Check format and convert if DICOM
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

        # Ensure we have 4 files by duplicating if fewer
        if len(nifti_files) < 4:
            st.warning("Fewer than 4 files detected. Duplicating some files to proceed.")
            while len(nifti_files) < 4 and len(nifti_files) > 0:
                nifti_files.append(nifti_files[-1])
        elif len(nifti_files) == 4:
            st.success("4 files have been successfully loaded.")
        elif len(nifti_files) > 4:
            st.warning("More than 4 files uploaded. Only the first 4 will be used.")

        # If "Preprocess" was clicked, run registration/rescale logic
        if preproc_clicked and len(nifti_files) > 0:
            st.info("Starting preprocessing (intensity rescale + registration)...")
            final_nifti_paths = []
            for item in nifti_files:
                if isinstance(item, str):
                    # Already a path
                    final_nifti_paths.append(item)
                else:
                    # UploadedFile => save to disk
                    item.seek(0)
                    tmp_path = os.path.join(os.getcwd(), item.name)
                    with open(tmp_path, "wb") as out:
                        out.write(item.read())
                    final_nifti_paths.append(tmp_path)

            preproc_paths = realign_images_to_reference(final_nifti_paths)
            st.success("Preprocessing completed. Generated files:")
            for p in preproc_paths:
                st.write(f"- {p}")

        # Show columns for each uploaded file (preview in tab1)
        st.write("## Preview of Uploaded Files in Columns")
        columns = st.columns(len(unique_files))
        for i, uf in enumerate(unique_files):
            with columns[i]:
                st.write(f"**{uf.name}**")
                uf.seek(0)
                fmt = check_file_format(uf)

                if fmt == "dicom":
                    uf.seek(0)
                    ds = pydicom.dcmread(uf)
                    volume = ds.pixel_array
                    if volume.ndim == 2:
                        # Make it 3D if only one slice
                        volume = volume[np.newaxis, ...]
                    max_slices = volume.shape[0] - 1
                    sidx = 0
                    if max_slices > 0:
                        sidx = st.slider(
                            f"Slice {uf.name}",
                            0, 
                            max_slices, 
                            0, 
                            key=f"dicom_slider_{i}"
                        )
                    # Insert a temporary dimension so show_simple expects [D,H,W]
                    # In this case, "volume" is already [D,H,W], so we can show directly 
                    fig = show_simple(volume, sidx, title=f"DICOM {uf.name}")
                    st.pyplot(fig)

                elif fmt == "nifti":
                    uf.seek(0)
                    tmp_path = os.path.join(os.getcwd(), uf.name)
                    with open(tmp_path, "wb") as out:
                        out.write(uf.read())
                    volume_data = nib.load(tmp_path).get_fdata()
                    if volume_data.ndim == 2:
                        volume_data = volume_data[np.newaxis, ...]
                    depth = volume_data.shape[0]
                    sidx = depth // 2
                    if depth > 1:
                        sidx = st.slider(
                            f"Slice {uf.name}",
                            0, 
                            depth - 1, 
                            sidx, 
                            key=f"nifti_slider_{i}"
                        )
                    fig = show_simple(volume_data, sidx, title=f"NIfTI {uf.name}")
                    st.pyplot(fig)
                else:
                    st.error("Unrecognized format for preview.")
    else:
        st.info("Please upload up to 4 files in NIfTI or DICOM format.")


# --------------------------------------------------------------------------------
# TAB 2: RESULTS
# --------------------------------------------------------------------------------
with results:
    # Two columns for predictions checkboxes
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predictions")
        show_ncr = st.checkbox("Show Necrotic Core")
        show_ed = st.checkbox("Show Edema")
        show_et = st.checkbox("Show Enhancing Tumor")

    with col2:
        st.subheader("Extras")
        # By default, the user must check to show GT
        show_reality = st.checkbox("Show Ground Truth", value=False)
        # Combined mask
        show_combined = st.checkbox("Show Combined Masks (NCR + Edema + ET)", value=True)

    # selected_maps will collect (map_name, map_path) pairs
    selected_maps = {}
    loaded_masks = {}

    # If "Show Ground Truth" is checked and a GT was uploaded, use it
    if show_reality and st.session_state.get("gt_path"):
        selected_maps["Ground Truth"] = st.session_state["gt_path"]

    # If other checkboxes are active, add their default paths
    if show_ncr:
        selected_maps["Necrotic Core"] = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_NCR.nii.gz"
        # selected_maps["Necrotic Core"] = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_NCR.nii.gz"
        selected_maps["Necrotic Core"] = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_NCR.nii.gz"
    if show_ed:
        selected_maps["Edema"] = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_ED.nii.gz"
        # selected_maps["Edema"] = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_ED.nii.gz"
        selected_maps["Edema"] = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_ED.nii.gz"
    if show_et:
        selected_maps["Enhancing Tumor"] = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_ET.nii.gz"
        # selected_maps["Enhancing Tumor"] = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_ET.nii.gz"
        selected_maps["Enhancing Tumor"] = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_ET.nii.gz"
    if show_combined:
        selected_maps["Combined Mask"] = "combined"

    # Prepare metrics dictionaries
    voxel_volume_mm3 = 1  # Adjust if voxel volume is known
    metrics = {}
    tumor_counts = {}
    tumor_volumes = {}
    masks_to_check = []

    # Load and compute
    for map_name, map_path in selected_maps.items():
        if map_name == "Combined Mask":
            # Combine the three maps (NCR, ED, ET)
            ncr_path = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_NCR.nii.gz"
            # ncr_path = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_NCR.nii.gz"
            ncr_path = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_NCR.nii.gz"
            ed_path = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_ED.nii.gz"
            # ed_path = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_ED.nii.gz"
            ed_path = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_ED.nii.gz"
            et_path = "./brats_example_segmentations/BraTS2021_00657/BraTS2021_00657_uncertainty_map_ET.nii.gz"
            # et_path = "./brats_example_segmentations/BraTS2021_00757/BraTS2021_00757_uncertainty_map_ET.nii.gz"
            et_path = "./brats_example_segmentations/BraTS2021_two_tumors_merged/BraTS2021_00688_AND_01309_uncertainty_map_ET.nii.gz"

            ncr = load_nifti(ncr_path)
            ed = load_nifti(ed_path)
            et = load_nifti(et_path)
            combined_mask = (ncr > 0)*1 + (ed > 0)*2 + (et > 0)*3

            loaded_masks[map_name] = combined_mask
            masks_to_check.append(combined_mask)

            # Calculate metrics for combined mask
            volume_voxels = (combined_mask > 0).sum()
            volume_mm3 = volume_voxels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000
            metrics[map_name] = volume_cm3

            tcount, tvols = count_tumors(combined_mask > 0)
            tumor_counts[map_name] = tcount
            tumor_volumes[map_name] = tvols

        else:
            # Load the mask from a known path
            data = load_nifti(map_path)
            loaded_masks[map_name] = data
            masks_to_check.append(data)

            # Calculate basic volume
            volume_voxels = (data > 0).sum()
            volume_mm3 = volume_voxels * voxel_volume_mm3
            volume_cm3 = volume_mm3 / 1000
            metrics[map_name] = volume_cm3

            # Tumor count
            tcount, tvols = count_tumors(data)
            tumor_counts[map_name] = tcount
            tumor_volumes[map_name] = tvols

    # If no maps selected, warn the user
    if len(loaded_masks) == 0:
        st.warning("No maps selected. Please select at least one option.")
    else:
        # Show tumor metrics
        st.subheader("Tumor Metrics")
        col_metrics = st.columns(len(loaded_masks))
        map_names_list = list(loaded_masks.keys())

        for idx, map_name in enumerate(map_names_list):
            # Check if we have metrics for this map
            if map_name in metrics:
                with col_metrics[idx]:
                    st.metric(label=f"{map_name} Tumors", value=f"{tumor_counts[map_name]}")
                    st.metric(label=f"{map_name} Total Volume", value=f"{metrics[map_name]:.2f}")
                    # If there's more than one tumor, show each volume
                    if tumor_counts[map_name] > 1:
                        for i, vol in enumerate(tumor_volumes[map_name]):
                            st.metric(label=f"Tumor {i+1} Volume", value=f"{vol:.2f}")

        # -------------------------------------------------------------------
        # Visualization: Each map in its own column, each with slice & threshold
        # -------------------------------------------------------------------
        st.subheader("Visualization per Map (slice and threshold)")
        col_visuals = st.columns(len(loaded_masks))

        for idx, map_name in enumerate(map_names_list):
            with col_visuals[idx]:
                volume_3d = loaded_masks[map_name]

                # 1) Find the non-empty slice range
                non_zero_slices = (volume_3d != 0).any(axis=(1,2)).nonzero()[0]
                if len(non_zero_slices) > 0:
                    min_sli = int(non_zero_slices[0])
                    max_sli = int(non_zero_slices[-1])
                else:
                    # If fully empty, range is the entire volume
                    min_sli = 0
                    max_sli = volume_3d.shape[0] - 1

                # Slider for slice selection
                slice_idx = st.slider(
                    f"{map_name} - Slice",
                    min_value=min_sli,
                    max_value=max_sli,
                    value=(min_sli + max_sli)//2,
                    key=f"slice_{map_name}"
                )

                # 2) Threshold slider
                #   We take min and max intensity in the volume
                vmin = float(volume_3d.min())
                vmax = float(volume_3d.max())
                step_val = (vmax - vmin)/100 if vmax>vmin else 1.0

                threshold = st.slider(
                    f"{map_name} - Threshold",
                    min_value=vmin,
                    max_value=vmax,
                    value=vmin,
                    step=step_val,
                    key=f"thr_{map_name}"
                )

                # 3) Apply threshold on the selected slice
                slice_2d = np.copy(volume_3d[slice_idx, :, :])
                slice_2d[slice_2d < threshold] = 0

                # "show_simple" expects a 3D array, so we add an axis
                slice_3d = slice_2d[np.newaxis, ...]  # shape => (1, H, W)

                fig = show_simple(slice_3d, 0, title=map_name)
                st.pyplot(fig)
