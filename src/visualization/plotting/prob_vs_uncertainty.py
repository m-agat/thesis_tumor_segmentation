import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

from ..utils.constants import GT_DIR, UNC_MAPS_DIR
from ..utils.io import load_nifti
from ..utils.plotting_helpers import find_slice_max_label, get_tumor_bounding_box

# map region names → segmentation labels & softmax channels
_LABEL_MAP = {
    "NCR": 1,
    "ED":  2,
    "ET":  3,
}

def plot_prob_vs_uncertainty(
    data_dirs: list[str],
    model_names: list[str],
    patient_id: str,
    sub_region: str = "ED",
    slice_idx: int | None = None,
    out_dir: str | None = None
):
    """
    Plot probability & uncertainty maps for a given sub-region
    (NCR, ED, or ET) side by side for multiple models.
    
    Args:
        data_dirs:   List of directories containing each model’s outputs
        model_names: List of model subfolder names (must match data_dirs order)
        patient_id:  e.g. "00332"
        sub_region:  One of "NCR", "ED", or "ET"
        slice_idx:   If None, auto-select slice via find_slice_max_label()
        out_dir:     Where to save; defaults to UNC_MAPS_DIR in config
    """
    # validate sub_region
    if sub_region not in _LABEL_MAP:
        raise ValueError(f"sub_region must be one of {_LABEL_MAP.keys()}")

    label = _LABEL_MAP[sub_region]

    # 1) prepare output directory
    out_dir = out_dir or str(UNC_MAPS_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # 2) load ground truth flair & seg volumes
    flair_path = os.path.join(
        GT_DIR, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_flair.nii.gz"
    )
    seg_path   = os.path.join(
        GT_DIR, f"BraTS2021_{patient_id}", f"BraTS2021_{patient_id}_seg.nii.gz"
    )
    flair_vol = load_nifti(flair_path)
    gt_seg    = load_nifti(seg_path)

    # 3) pick slice if not provided
    if slice_idx is None:
        slice_idx, _ = find_slice_max_label(
            patient_id, primary_label=label
        )

    # 4) compute bounding box around that region
    region_mask = (gt_seg == label).astype(int)
    xmin, xmax, ymin, ymax = get_tumor_bounding_box(
        region_mask, slice_idx, padding=20
    )

    # 5) set up figure/GridSpec
    n = len(model_names)
    fig = plt.figure(figsize=(12, 3 * n))
    gs  = GridSpec(
        n + 1, 6,
        width_ratios =[0.3, 1, 1, 1, 1, 0.1],
        height_ratios=[0.3] + [1]*n,
        hspace=0.15,
        wspace=0.05
    )

    # 6) segmentation overlay colormap
    cmap_seg = ListedColormap(['black','red','yellow','blue'])

    # 7) header row
    headers = ["Model",
               f"{sub_region} Prob.",
               f"{sub_region} Unc.",
               "Pred.",
               "Ground Truth"]
    for col, title in enumerate(headers):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title,
                ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax.axis('off')

    # 8) plot each model’s row
    for row, (mname, ddir) in enumerate(zip(model_names, data_dirs), start=1):
        display = mname.upper() if mname!='hybrid_new' else 'Hybrid'

        # build file paths
        pred_seg_path = os.path.join(ddir, f"seg_{patient_id}.nii.gz")
        softmax_path  = os.path.join(ddir, f"softmax_{patient_id}.nii.gz")
        unc_path      = os.path.join(ddir, f"uncertainty_{sub_region}_{patient_id}.nii.gz")

        # extract slices
        flair_slice = flair_vol[:, :, slice_idx]
        pred_slice  = load_nifti(pred_seg_path)[:, :, slice_idx]
        prob_vol    = load_nifti(softmax_path)
        prob_slice  = prob_vol[label, :, :, slice_idx]
        unc_slice   = load_nifti(unc_path)[:, :, slice_idx]
        gt_slice    = gt_seg[:, :, slice_idx]

        # a) model name cell
        ax = fig.add_subplot(gs[row, 0])
        ax.text(0.5, 0.5, display,
                ha='center', va='center',
                fontsize=11, fontweight='bold')
        ax.axis('off')

        # b) probability
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(prob_slice, cmap='hot', alpha=0.5)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # c) uncertainty
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(flair_slice, cmap='gray')
        im = ax.imshow(unc_slice, cmap='hot', alpha=0.5,
                       vmin=unc_slice.min(), vmax=unc_slice.max())
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # d) prediction overlay
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(pred_slice, cmap=cmap_seg, alpha=0.5,
                  vmin=0, vmax=3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
        ax.axis('off')

        # e) ground truth
        ax = fig.add_subplot(gs[row, 4])
        ax.imshow(flair_slice, cmap='gray')
        ax.imshow(gt_slice, cmap=cmap_seg, alpha=0.5,
                  vmin=0, vmax=3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymax, ymin)
        ax.axis('off')

    # 9) colorbar on the right
    cax = fig.add_subplot(gs[1:,5])
    cb  = plt.colorbar(im, cax=cax)
    cb.set_label('Uncertainty / Probability',
                 fontsize=11, fontweight='bold')

    # 10) title, save & close
    plt.suptitle(f"Patient {patient_id} — {sub_region} slice {slice_idx}",
                 fontsize=14, fontweight='bold', y=0.92)
    out_path = os.path.join(out_dir,
                            f"prob_vs_unc_{sub_region}_{patient_id}_slice{slice_idx}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close(fig)

    return out_path
