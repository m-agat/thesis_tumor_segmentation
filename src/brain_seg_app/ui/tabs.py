import os
import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd 
import torch
from file_utils import find_available_outputs
from visualization import (
    plot_segmentation,
    plot_ground_truth,
    plot_probabilities,
    plot_uncertainty,
    volumes_from_nii
)
from evaluate_performance import compute_metrics


def render_data_tab(raw_paths: list[str], preproc_paths: list[str] | None):
    st.header("Welcome to the Brain Tumor Segmentation App! 👋")
    st.markdown(
        """
        **What does this app do?**  
        - Upload up to 4 MRI scans (NIfTI or DICOM).  
        - (Optional) Preprocess: skull strip, reslice, register.  
        - Run ensemble model for tumor segmentation.  
        """
    )

    # Original scans
    st.subheader("🧠 Original Scans")
    if raw_paths:
        img0 = nib.load(raw_paths[0])
        max_slice = img0.shape[-1]
        idx = st.slider("Slice Index (Original)", 0, max_slice - 1, max_slice // 2, key="orig_slice")
        cols = st.columns(len(raw_paths))
        for col, path in zip(cols, raw_paths):
            modality = extract_modality(path)
            img   = nib.load(path, mmap=False)
            data  = img.get_fdata()
            slice_img = data[..., idx]
            with col:
                st.markdown(f"**{modality}**", unsafe_allow_html=True)
                vmin, vmax = np.percentile(slice_img, (1, 99))
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(slice_img, cmap="gray", vmin=vmin, vmax=vmax)
                ax.axis("off")
                st.pyplot(fig)
    else:
        st.warning("No original scans to preview.")

    # Preprocessed scans
    if preproc_paths and preproc_paths != raw_paths and all(os.path.exists(p) for p in preproc_paths):
        st.subheader("🧼 Preprocessed Scans")
        img0 = nib.load(preproc_paths[0])
        max_slice = img0.shape[-1]
        idx2 = st.slider("Slice Index (Preprocessed)", 0, max_slice - 1, max_slice // 2, key="preproc_slice")
        cols2 = st.columns(len(preproc_paths))
        for col, path in zip(cols2, preproc_paths):
            img   = nib.load(path, mmap=False)
            data  = img.get_fdata()
            slice_img = data[..., idx]
            with col:
               vmin, vmax = np.percentile(slice_img, (1, 99))
               fig, ax = plt.subplots(figsize=(3,3))
               ax.imshow(slice_img, cmap="gray", vmin=vmin, vmax=vmax)
               ax.axis("off")
               st.pyplot(fig)
    elif preproc_paths:
        st.info("No new preprocessed scans; previewing original scans only.")

def extract_modality(path: str) -> str:
    basename = os.path.basename(path).lower()
    for mod in ["flair", "t1ce", "t1", "t2"]:
        if mod in basename:
            return mod.upper() if mod != "t1ce" else "T1ce"
    return "Unknown"

def render_results_tab(
    output_dir: str,
    raw_paths: list[str] | None,
    preproc_paths: list[str] | None,
    gt_path: str | None
):
    st.header("Results & Visualization")
    outputs = find_available_outputs(output_dir)
    if not outputs:
        st.warning("No segmentations found. Please run segmentation first.")
        return

    # Patient selection
    pid = st.selectbox("Select Patient ID", sorted(outputs.keys()), key="res_pid")

    # Selection panels
    col_a, col_b = st.columns(2)
    with col_a:
        show_seg = st.checkbox("Show Predicted Segmentation", value=True, key="show_seg")
        seg_tissues = []
        if show_seg:
            with st.expander("Select Segmentation Tissues", expanded=True):
                if st.checkbox("NCR (Label 1)", key=f"seg_ncr_{pid}"):
                    seg_tissues.append((1, (1, 0, 0)))
                if st.checkbox("ED (Label 2)", key=f"seg_ed_{pid}"):
                    seg_tissues.append((2, (0, 1, 0)))
                if st.checkbox("ET (Label 3)", key=f"seg_et_{pid}"):
                    seg_tissues.append((3, (0, 0, 1)))
    with col_b:
        show_gt = False
        gt_tissues = []
        if gt_path or outputs[pid].get("gt"):
            show_gt = st.checkbox("Show Ground Truth", value=False, key="show_gt")
        if show_gt:
            with st.expander("Select Ground Truth Tissues", expanded=True):
                if st.checkbox("NCR (Label 1)", key=f"gt_ncr_{pid}"):
                    gt_tissues.append((1, (1, 0, 0)))
                if st.checkbox("ED (Label 2)", key=f"gt_ed_{pid}"):
                    gt_tissues.append((2, (0, 1, 0)))
                if st.checkbox("ET (Label 3)", key=f"gt_et_{pid}"):
                    gt_tissues.append((3, (0, 0, 1)))

    col_c, col_d = st.columns(2)
    with col_c:
        show_prob = st.checkbox("Show Probability Map", value=False, key="show_prob")
        prob_slices = []
        if show_prob:
            with st.expander("Select Probability Tissues", expanded=True):
                soft_path = outputs[pid].get("softmax")
                if soft_path:
                    soft = nib.load(soft_path).get_fdata()
                    if st.checkbox("NCR (Label 1)", key=f"prob_ncr_{pid}"):
                        prob_slices.append(("NCR", soft[1]))
                    if st.checkbox("ED (Label 2)", key=f"prob_ed_{pid}"):
                        prob_slices.append(("ED", soft[2]))
                    if st.checkbox("ET (Label 3)", key=f"prob_et_{pid}"):
                        prob_slices.append(("ET", soft[3]))
    with col_d:
        show_unc = st.checkbox("Show Uncertainty Map", value=False, key="show_unc")
        unc_slices = []
        thr, mode = 0.5, "Below"
        if show_unc:
            with st.expander("Select Uncertainty Tissues", expanded=True):
                if st.checkbox("NCR (Label 1)", key=f"unc_ncr_{pid}"):
                    unc_slices.append(("NCR", nib.load(outputs[pid]["uncertainty_NCR"]).get_fdata()))
                if st.checkbox("ED (Label 2)", key=f"unc_ed_{pid}"):
                    unc_slices.append(("ED", nib.load(outputs[pid]["uncertainty_ED"]).get_fdata()))
                if st.checkbox("ET (Label 3)", key=f"unc_et_{pid}"):
                    unc_slices.append(("ET", nib.load(outputs[pid]["uncertainty_ET"]).get_fdata()))
                mode = st.radio("Threshold Mode", ["Below", "Above"], horizontal=True, key=f"thr_mode_{pid}")
                thr = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.5, step=0.01, key=f"thr_{pid}")

    # Modality selection (default Flair)
    scan_paths = preproc_paths if preproc_paths else raw_paths
    brain_full = None
    if scan_paths:
        with st.expander("Select Modality for Overlay", expanded=True):
            modalities = ["Flair", "T1ce", "T1", "T2"]
            sel_mod = st.radio("Modality", modalities, index=0, key="overlay_mod")
        mod_map = dict(zip(modalities, scan_paths))
        brain_full = nib.load(mod_map[sel_mod]).get_fdata()

    # Slice index
    sample = brain_full if brain_full is not None else nib.load(outputs[pid].get("seg")).get_fdata()
    max_slice = sample.shape[-1]
    slice_idx = st.slider("Slice Index", 0, max_slice - 1, max_slice // 2, key="res_slice")
    brain_slice = (brain_full[..., slice_idx] if brain_full is not None else sample[..., slice_idx])

    # Opacity control (used for segmentation, probability, uncertainty overlays)
    opacity = st.slider("Overlay Opacity", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="overlay_opacity")

    fig_w, fig_h = 600, 600

    # Row 1: Segmentation & GT
    cols1 = st.columns(2, gap="small")
    # Predicted Segmentation
    with cols1[0]:
        st.subheader("Predicted Segmentation")
        if show_seg:
            fig_seg = plot_segmentation(
                brain_slice,
                nib.load(outputs[pid]["seg"]).get_fdata()[..., slice_idx],
                seg_tissues,
                opacity=opacity,
                width=fig_w,
                height=fig_h
            )
            st.plotly_chart(fig_seg, use_container_width=True, key=f"seg_fig_{pid}")
            # After your segmentation chart:
            buf_seg = fig_seg.to_image(format="png")
            st.download_button(
                "Download Segmentation Figure",
                data=buf_seg,
                file_name=f"segmentation_{pid}.png",
                mime="image/png",
            )

        else:
            st.write("Segmentation hidden. Enable checkbox above.")
    # Ground Truth
    with cols1[1]:
        st.subheader("Ground Truth")
        if show_gt:
            fig_gt = plot_ground_truth(
                brain_slice,
                nib.load(gt_path or outputs[pid].get("gt")).get_fdata()[..., slice_idx],
                gt_tissues,
                opacity=opacity,
                width=fig_w,
                height=fig_h
            )
            st.plotly_chart(fig_gt, use_container_width=True, key=f"gt_fig_{pid}")
        else:
            st.write("Ground truth hidden. Enable checkbox above.")

    # Row 2: Probability & Uncertainty
    cols2 = st.columns(2, gap="small")
    with cols2[0]:
        st.subheader("Probability Map")
        if show_prob:
            fig_prob = plot_probabilities(
                brain_slice,
                [(lbl, data[..., slice_idx]) for lbl, data in prob_slices],
                opacity=opacity,
                width=fig_w,
                height=fig_h
            )
            st.plotly_chart(fig_prob, use_container_width=True, key=f"prob_fig_{pid}")
            buf_prob = fig_prob.to_image(format="png")
            st.download_button(
                "Download Probability Map",
                data=buf_prob,
                file_name=f"probability_{pid}.png",
                mime="image/png",
            )
        else:
            st.write("Probability map hidden. Enable checkbox above.")
    with cols2[1]:
        st.subheader("Uncertainty Map")
        if show_unc:
            fig_unc = plot_uncertainty(
                brain_slice,
                [(lbl, data[..., slice_idx]) for lbl, data in unc_slices],
                threshold=thr,
                mode=mode,
                opacity=opacity,
                cmap="hot",
                width=fig_w,
                height=fig_h
            )
            st.plotly_chart(fig_unc, use_container_width=True, key=f"unc_fig_{pid}")
            buf_unc = fig_unc.to_image(format="png")
            st.download_button(
                "Download Uncertainty Map",
                data=buf_unc,
                file_name=f"uncertainty_{pid}.png",
                mime="image/png",
            )
        else:
            st.write("Uncertainty map hidden. Enable checkbox above.")

    # ─── VOLUMES ────────────────────────────────────────────────────────────────
    pred_vols = volumes_from_nii(outputs[pid]["seg"])       # dict
    gt_vols   = {}
    if show_gt:
        gt_path_eff = gt_path or outputs[pid].get("gt")
        if gt_path_eff:
            gt_vols = volumes_from_nii(gt_path_eff)

    # ---------- predicted -------------------------------------------------------

    st.markdown("## 📏 **Tumour Volumes**")
    st.subheader("Predicted")

    labels = ["NCR (cm³)", "ED (cm³)", "ET (cm³)", "Total (cm³)"]
    default_dict = {lbl: 0.0 for lbl in labels}
    pred_dict = {**default_dict, **pred_vols}

    c1, c2, c3, c4 = st.columns(4)

    for col, lbl in zip((c1, c2, c3, c4), labels):
        value = pred_dict[lbl]
        col.markdown(
            f"""<div style="text-align:center;">
                    <h4 style="margin-bottom:0.2rem;">{lbl}</h4>
                    <h2 style="margin-top:0;">{value:.2f}</h2>
                </div>""",
            unsafe_allow_html=True,
        )

    if show_gt:
        st.subheader("Ground truth")
        gt_dict = {**default_dict, **gt_vols}

        c1, c2, c3, c4 = st.columns(4)

        for col, lbl in zip((c1, c2, c3, c4), labels):
            value = gt_dict[lbl]
            col.markdown(
                f"""<div style="text-align:center;">
                        <h4 style="margin-bottom:0.2rem;">{lbl}</h4>
                        <h2 style="margin-top:0;">{value:.2f}</h2>
                    </div>""",
                unsafe_allow_html=True,
            )
    
    # ─── PERFORMANCE METRICS ───────────────────────────────────────────────────
    if show_gt and outputs[pid].get("seg") and (gt_path or outputs[pid].get("gt")):
        st.markdown("## 📊 **Segmentation Performance Metrics**")

        # load raw arrays
        seg_arr = nib.load(outputs[pid]["seg"]).get_fdata().astype(np.uint8)
        gt_arr  = nib.load(gt_path or outputs[pid].get("gt")).get_fdata().astype(np.uint8)

        # build binary masks per label (1=NCR, 2=ED, 3=ET)
        labels = [1, 2, 3]
        pred_list = [torch.from_numpy((seg_arr == lbl).astype(np.uint8)) for lbl in labels]
        gt_list   = [torch.from_numpy((gt_arr  == lbl).astype(np.uint8)) for lbl in labels]

        # compute your metrics
        dice_scores, hd95_scores, sens_scores, spec_scores = compute_metrics(pred_list, gt_list)

        # put into a pandas table
        df = pd.DataFrame({
            "Sub-region": ["NCR", "ED", "ET"],
            "Dice":       dice_scores,
            "HD95 (mm)":  hd95_scores,
            "Sensitivity": sens_scores,
            "Specificity": spec_scores
        })

        # display
        st.table(df)
