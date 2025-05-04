import re
from pathlib import Path
import numpy as np
import nibabel as nib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from visualization.utils.constants import GT_DIR  # root folder for ground‐truth volumes

def gather_seg_pairs(
    pred_dir: Path,
    model: str,
    labels: list[int] = [0,1,2,3]
) -> list[tuple[Path,Path]]:
    """
    Return list of (gt_path, pred_path) for every patient that has both.
    """
    pred_files = sorted(p for p in pred_dir.iterdir()
                        if re.fullmatch(fr"seg_{model}_\d{{5}}\.nii\.gz", p.name))
    pairs = []
    for pf in pred_files:
        pid = re.search(rf"seg_{model}_(\d{{5}})", pf.name).group(1)
        gt = Path(GT_DIR)/f"BraTS2021_{pid}"/f"BraTS2021_{pid}_seg.nii.gz"
        if gt.exists():
            pairs.append((gt, pf))
        else:
            print(f"  ⚠️  missing GT for {pid}, skipping")
    return pairs

def compute_aggregated_confusion(
    pairs: list[tuple[Path,Path]],
    labels: list[int] = [0,1,2,3]
) -> np.ndarray:
    """
    Given list of (gt_path, pred_path), load volumes, flatten, sum per-patient CMs.
    """
    agg = np.zeros((len(labels),len(labels)), dtype=np.int64)
    for gt_path, pred_path in pairs:
        gt = nib.load(str(gt_path)).get_fdata().astype(int).flatten()
        pr = nib.load(str(pred_path)).get_fdata().astype(int).flatten()
        if gt.shape != pr.shape:
            print(f"  ⚠️  shape mismatch {gt_path.name}")
            continue
        agg += confusion_matrix(gt, pr, labels=labels)
    return agg

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[int] = [0,1,2,3],
    class_names: list[str]=["BG","NCR","ED","ET"],
    cmap_colors: list[str]=None,
    out_path: Path = Path("confusion_matrix.png")
):
    """
    Normalize, convert to %, and seaborn‑plot the heatmap, then save.
    """
    # normalize rows → fractions
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)
    pct  = (norm * 100).round(1)

    # colormap
    if cmap_colors is None:
        cmap_colors = ['#f7fcfd','#e0ecf4','#bfd3e6','#9ebcda',
                       '#8c96c6','#8c6bb1','#88419d','#6e016b']
    cmap = mpl.colors.LinearSegmentedColormap.from_list("custom", cmap_colors)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        pct, annot=True, fmt=".1f", cmap=cmap,
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=100
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (%)")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(out_path), dpi=300)
    plt.close()

def main():
    import argparse
    p = argparse.ArgumentParser(description="Aggregate & plot 3D confusion matrix")
    p.add_argument("model", help="model subfolder name (e.g. vnet, simple_avg)")
    p.add_argument("--pred-dir", "-p", type=Path,
                   default=Path(__file__).parents[3]/"models"/"predictions"/"{model}",
                   help="where to find {model}_{pid}.nii.gz")
    p.add_argument("--out", "-o", type=Path, default=Path("confusion_matrices"),
                   help="where to save PNG")
    args = p.parse_args()

    preds = Path(args.pred_dir)
    pairs = gather_seg_pairs(preds, args.model)
    agg   = compute_aggregated_confusion(pairs)
    outfn = args.out/f"{args.model}_confusion_matrix.png"
    plot_confusion_matrix(agg, out_path=outfn)
    print("Saved to", outfn)

if __name__=="__main__":
    main()
