import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from pathlib import Path

from ..utils.constants import GT_DIR, CASE_SLICES, CASES, SOURCES_ENSEMBLE, FIGURES_DIR, ENSEMBLE_OUT, TITLE_FONTSIZE
from ..utils.io import load_nifti
from ..utils.plotting_helpers import find_slice_max_label, get_tumor_bounding_box, crop_to_bbox

# Ensure GT_DIR and FIGURES_DIR are Path objects
GT_DIR = Path(GT_DIR)
FIGURES_DIR = Path(FIGURES_DIR)

def plot_important_cases(
    patient_list: list[tuple[str,str]] = CASES,
    model_sources: list[tuple[str,str]] = SOURCES_ENSEMBLE,
    out_path: str | None = None
):
    """
    Plot a grid of important cases with ground truth overlays, ensuring
    all crops are the same size.

    patient_list: list of tuples (patient_id, title)
    model_sources: list of tuples (model_dir, display_name)
    out_path: where to save figure
    """
    # overlay colormap for segmentation labels
    cmap_seg = ListedColormap(['black','red','yellow','blue'])

    # First pass: compute bounding boxes for each patient to get uniform crop size
    bboxes = {}
    heights = []
    widths = []
    for pid, _ in patient_list:
        # load segmentation volume
        seg_path = GT_DIR / f"BraTS2021_{pid}" / f"BraTS2021_{pid}_seg.nii.gz"
        seg_vol = load_nifti(str(seg_path))
        # determine slice
        slice_idx = CASE_SLICES.get(pid)
        if slice_idx is None:
            slice_idx, _ = find_slice_max_label(pid)
        # compute bbox
        xmin, xmax, ymin, ymax = get_tumor_bounding_box(seg_vol, slice_idx, padding=20)
        bboxes[pid] = (ymin, ymax, xmin, xmax)
        heights.append(ymax - ymin)
        widths.append(xmax - xmin)
    max_h = max(heights)
    max_w = max(widths)

    # figure grid dimensions
    n_rows = len(model_sources)
    n_cols = len(patient_list) + 1  # extra column for model names
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, wspace=0.1, hspace=0.1)

    for r, (src, src_label) in enumerate(model_sources):
        # leftmost: model name
        ax = fig.add_subplot(gs[r, 0])
        ax.text(0.5, 0.5, src_label,
                ha='center', va='center', fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax.axis('off')

        for c, (pid, title) in enumerate(patient_list, start=1):
            ymin, ymax, xmin, xmax = bboxes[pid]

            # load FLAIR
            flair_vol = load_nifti(str(
                GT_DIR / f"BraTS2021_{pid}" / f"BraTS2021_{pid}_t1ce.nii.gz"
            ))

            if src.lower() == 'gt':
                # ground truth segmentation lives under GT_DIR
                gt_path = GT_DIR / f"BraTS2021_{pid}" / f"BraTS2021_{pid}_seg.nii.gz"
                pred_vol = load_nifti(str(gt_path))
            else:
                # ensemble model prediction under ENSEMBLE_OUT
                pred_path = ENSEMBLE_OUT / src / f"seg_{pid}.nii.gz"
                pred_vol  = load_nifti(str(pred_path))

            # pick slice
            slice_idx = CASE_SLICES.get(pid) or find_slice_max_label(pid)[0]

            # crop
            img_crop  = crop_to_bbox(flair_vol[:, :, slice_idx], (ymin, ymax, xmin, xmax))
            seg_crop  = crop_to_bbox(pred_vol[:, :, slice_idx].astype(int),
                                     (ymin, ymax, xmin, xmax))

            # pad to max_h, max_w
            pad_img = np.zeros((max_h, max_w), dtype=img_crop.dtype)
            pad_seg = np.zeros((max_h, max_w), dtype=seg_crop.dtype)
            h, w = img_crop.shape
            pad_img[:h, :w] = img_crop
            pad_seg[:h, :w] = seg_crop

            ax = fig.add_subplot(gs[r, c])
            ax.imshow(pad_img, cmap='gray', interpolation='none', aspect='auto')
            ax.imshow(pad_seg, cmap=cmap_seg, alpha=0.5, interpolation='none', aspect='auto')
            if r == 0:
                ax.set_title(f"Patient {pid}", fontsize=TITLE_FONTSIZE, pad=6, fontweight="bold")
            ax.axis('off')

    fig.tight_layout(pad=0.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04, wspace=0.02, hspace=0.02)
    out = Path(out_path) if out_path else (FIGURES_DIR / "important_cases.png")
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return str(out)
