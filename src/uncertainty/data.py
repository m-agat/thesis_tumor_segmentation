from pathlib import Path
import pandas as pd
import nibabel as nib
import numpy as np
from typing import Dict, Iterator, Tuple
import re 

def load_ece_results(model_name: str, csv_path: Path) -> pd.DataFrame:
    """
    Read per-subregion ECE results for one model.
    Assumes columns like 'NCR_ECE', 'ED_ECE', 'ET_ECE'.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.upper()  # normalize
    df["MODEL"] = model_name
    return df

def load_all_models_ece(model_csvs: Dict[str, Path]) -> pd.DataFrame:
    """
    Given a mapping model_name→csv_path, 
    return a single DataFrame with a 'MODEL' column.
    """
    dfs = [load_ece_results(name, path) for name, path in model_csvs.items()]
    return pd.concat(dfs, ignore_index=True)

def list_files(base_dir: Path, pattern: str) -> Iterator[Path]:
    """Yield all files in base_dir matching a glob pattern."""
    yield from sorted(base_dir.glob(pattern))

def load_region_maps(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str,
    gt_pattern: str,
    subregions: Dict[str,int],
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict[str,np.ndarray]]]:
    """
    For each patient in `pred_dir` matching `pred_pattern`, load:
      - softmax volume (CxHxWxD)
      - GT seg volume (HxWxD)
      - any per-region uncertainty maps under pred_dir matching UNCERTAINTY_PATTERN
    Yields (patient_id, gt_vol, prob_vol, {region:unc_vol})
    """
    from .constants import UNCERTAINTY_PATTERN
    # 1) list your softmax files
    softmax_files = sorted(pred_dir.glob(pred_pattern))

    for fp in softmax_files:
        m = re.match(r".*_(\d+)\.nii\.gz$", fp.name)
        if not m:
            print(f"[WARN] skipping file with unexpected name: {fp.name}")
            continue
        pid_num = m.group(1)                # e.g. "00332"
        pid     = f"BraTS2021_{pid_num}"    # e.g. "BraTS2021_00332"

        gt_glob = gt_pattern.replace("{id}", pid)
        gt_fp = next(gt_dir.glob(gt_glob), None)
        if gt_fp is None:
            continue

        gt_vol = nib.load(gt_fp).get_fdata().astype(int)
        prob_vol = nib.load(fp).get_fdata()

        # find all uncertainty maps matching the global UNCERTAINTY_PATTERN,
        # then pick out only those for this patient & each region
        uncs = {}
        for region, label in subregions.items():
            pat = UNCERTAINTY_PATTERN.replace("*", f"*_{region}_{pid}*")
            up = next(pred_dir.glob(pat), None)
            uncs[region] = (nib.load(up).get_fdata() if up else np.zeros_like(gt_vol))

        yield pid, gt_vol, prob_vol, uncs

def load_error_uncertainty_data(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str,
    unc_patterns: Dict[str,str],
    gt_pattern: str,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Yields (patient_id, gt_volume, prob_volume, {region:uncertainty_volume})
    for every case in pred_dir matching pred_pattern.
    unc_patterns maps region→glob pattern for its uncertainty map.
    """
    # list base files
    softmax_files = sorted(pred_dir.glob(pred_pattern))
    for prob_fp in softmax_files:
        m = re.match(r".*_(\d+)\.nii\.gz$", prob_fp.name)
        if not m:
            print(f"[WARN] skipping file with unexpected name: {prob_fp.name}")
            continue
        pid_num = m.group(1)                # e.g. "00332"
        pid     = f"BraTS2021_{pid_num}"    # e.g. "BraTS2021_00332"

        # load GT
        gt_glob = gt_pattern.replace("{id}", pid)
        gt_fp = next(gt_dir.glob(gt_glob), None)
        if gt_fp is None:
            continue
        gt = nib.load(gt_fp).get_fdata().astype(int)
        prob = nib.load(prob_fp).get_fdata()
        uncs = {}
        for region, upat in unc_patterns.items():
            upath = next(pred_dir.glob(upat.replace("{id}", pid)), None)
            uncs[region] = (nib.load(upath).get_fdata() if upath else np.zeros_like(gt))
        
        yield pid, gt, prob, uncs

def load_conf_correct_by_class(
    pred_dir: Path,
    gt_dir: Path,
    pred_pattern: str,
    gt_pattern: str,
    class_labels: Dict[int,str]
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Yields (class_index, all_confidences, all_correctness) aggregated
    across *all* patients for that class.
    """
    # initialize buffers
    buffers = {c: {"conf": [], "corr": []} for c in class_labels}

    # find all prediction files
    for prob_fp in sorted(pred_dir.glob(pred_pattern)):
        pid = prob_fp.stem.split("_")[-1]
        gt_fp = next(gt_dir.glob(gt_pattern.replace("{id}", pid)), None)
        if gt_fp is None:
            continue

        # load volumes
        gt = nib.load(gt_fp).get_fdata().astype(int)
        prob = nib.load(prob_fp).get_fdata()  # shape (C, H, W, D)

        for c in class_labels:
            conf_c = prob[c].ravel()
            corr_c = (gt.ravel() == c).astype(np.float32)
            buffers[c]["conf"].append(conf_c)
            buffers[c]["corr"].append(corr_c)

    # concatenate and yield
    for c, data in buffers.items():
        if not data["conf"]:
            continue
        yield (
            c,
            np.concatenate(data["conf"]),
            np.concatenate(data["corr"])
        )