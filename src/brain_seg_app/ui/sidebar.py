import os
import glob
import streamlit as st
from config_web import BASE_DIR, RAW_UPLOADS
from file_utils import check_file_format, convert_dicom_to_nifti, reorder_modalities


def file_upload_section():
    """
    Displays the sidebar UI for uploading MRI scans and optional ground truth.

    Returns:
        raw_paths (list[str]): List of local file paths to up to 4 NIfTI scans.
        gt_path    (str or None):  Local file path to uploaded ground truth NIfTI, or None.
    """
    st.sidebar.header("File Upload")
    use_example = st.sidebar.checkbox(
        "Use Example Case", value=False,
        help="Use pre-packaged sample images (in the assets/example/raw folder)"
    )

    if use_example:
        example_dir = os.path.join(BASE_DIR, "assets", "example", "raw")
        st.sidebar.info("Using example case.")
        raw_paths = glob.glob(os.path.join(example_dir, "*.nii*"))
        raw_paths = reorder_modalities(raw_paths)
    else:
        uploaded = st.sidebar.file_uploader(
            "Upload up to 4 files (NIfTI or DICOM)",
            type=["nii", "nii.gz", "dcm"],
            accept_multiple_files=True
        )
        raw_paths = []
        if not uploaded:
            st.sidebar.warning("Please upload files or select Example Case.")
        else:
            # Deduplicate by filename
            seen = set()
            unique = []
            for f in uploaded:
                if f.name not in seen:
                    seen.add(f.name)
                    unique.append(f)

            # Process each file: detect format and convert if needed
            os.makedirs(RAW_UPLOADS, exist_ok=True)
            for f in unique:
                fmt = check_file_format(f)
                if fmt == "nifti":
                    st.sidebar.success(f"NIfTI: {f.name}")
                    out_path = os.path.join(RAW_UPLOADS, f.name)
                    with open(out_path, "wb") as buf:
                        buf.write(f.read())
                    raw_paths.append(out_path)
                elif fmt == "dicom":
                    st.sidebar.warning(f"DICOM: {f.name} → converting...")
                    f.seek(0)
                    out_name = f"converted_{os.path.splitext(f.name)[0]}.nii"
                    nifti_path = convert_dicom_to_nifti([f], RAW_UPLOADS, out_name)
                    if nifti_path:
                        st.sidebar.info(f"Converted to NIfTI: {os.path.basename(nifti_path)}")
                        raw_paths.append(nifti_path)
                else:
                    st.sidebar.error(f"Unrecognized format: {f.name}")

            # Ensure exactly 4 scans: pad or truncate
            if len(raw_paths) < 4 and raw_paths:
                st.sidebar.warning("Fewer than 4 scans: duplicating last to fill.")
                while len(raw_paths) < 4:
                    raw_paths.append(raw_paths[-1])
            elif len(raw_paths) > 4:
                st.sidebar.warning("More than 4 scans: using first 4.")
                raw_paths = raw_paths[:4]

            # Reorder into Flair, T1ce, T1, T2
            raw_paths = reorder_modalities(raw_paths)

    # Optional ground truth upload
    gt_path = None
    gt_file = st.sidebar.file_uploader(
        "Upload Ground Truth (optional)",
        type=["nii", "nii.gz"]
    )
    if gt_file is not None:
        gt_dest = os.path.join(BASE_DIR, os.path.basename(gt_file.name))
        with open(gt_dest, "wb") as buf:
            buf.write(gt_file.read())
        st.sidebar.success(f"Loaded GT: {gt_file.name}")
        gt_path = gt_dest

    return raw_paths, gt_path
