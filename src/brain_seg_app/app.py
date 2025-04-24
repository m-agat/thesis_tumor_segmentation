import streamlit as st

from config_web import PAGE_TITLE, LAYOUT, PREPROC_DIR, SEG_DIR
from ui.sidebar import file_upload_section
from ui.tabs import render_data_tab, render_results_tab
from data_loader import create_test_loader
from preprocessing import realign_images_to_reference
from segmentation import load_models, run_ensemble


def main():
    # Page config
    st.set_page_config(page_title=PAGE_TITLE, layout=LAYOUT)
    st.title(PAGE_TITLE)

    # Sidebar: file upload and ground truth
    raw_paths, gt_path = file_upload_section()

    # Preprocess button
    if st.sidebar.button("Preprocess", key="btn_preprocess"):
        if not raw_paths:
            st.sidebar.error("No scans to preprocess.")
        else:
            st.sidebar.info("Starting preprocessing...")
            prog = st.sidebar.progress(0)
            # callback to update sidebar progress
            def progress_cb(pct: float):
                prog.progress(int(pct * 100))

            preproc_paths = realign_images_to_reference(
                raw_paths,
                PREPROC_DIR,
                progress_callback=progress_cb
            )
            st.session_state["preproc_paths"] = preproc_paths
            prog.empty()
            st.sidebar.success("Preprocessing complete.")

    # Segment button
    if st.sidebar.button("Run Tumor Segmentation", key="btn_segment"):
        paths = st.session_state.get("preproc_paths", raw_paths)
        if not paths:
            st.sidebar.error("No data available for segmentation.")
        else:
            st.sidebar.info("Loading ensemble models...")
            models = load_models()

            st.sidebar.info("Preparing data loader...")
            loader = create_test_loader(paths)

            st.sidebar.info("Running segmentation...")
            prog2 = st.sidebar.progress(0)
            run_ensemble(
                loader,
                models,
                SEG_DIR,
                progress_bar=prog2
            )
            prog2.empty()
            st.sidebar.success("Segmentation complete.")

    # Main tabs
    tab1, tab2 = st.tabs(["Data Upload & Preview", "Results"])
    with tab1:
        preproc = st.session_state.get("preproc_paths")
        render_data_tab(raw_paths, preproc)

    with tab2:
        preproc = st.session_state.get("preproc_paths")
        render_results_tab(
            output_dir=SEG_DIR,
            raw_paths=raw_paths,
            preproc_paths=preproc,
            gt_path=gt_path
        )


if __name__ == "__main__":
    main()
