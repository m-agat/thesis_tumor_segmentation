import os
import json
import pandas as pd
import numpy as np
import nibabel as nib

from utils.metrics import compute_metrics

REGEX_PATIENT = r"seg_(?P<pid>\w+)\.nii\.gz"


def evaluate_patient(pred_path: str, gt_path: str) -> dict:
    """
    Evaluate one patient: load pred and gt NIfTIs, compute metrics.
    Returns a dict of metrics.
    """
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt   = nib.load(gt_path).get_fdata().astype(np.uint8)

    # assume labels in {0,1,2,...}
    pred_list = [ (pred == i).astype(np.uint8) for i in np.unique(np.concatenate([pred, gt])) ]
    gt_list   = [ (gt   == i).astype(np.uint8) for i in np.unique(np.concatenate([pred, gt])) ]

    dice, hd95, sens, spec = compute_metrics(pred_list, gt_list)

    # extract patient ID from filename
    import re
    m = re.search(REGEX_PATIENT, os.path.basename(pred_path))
    pid = m.group('pid') if m else os.path.splitext(os.path.basename(pred_path))[0]

    metrics = {'patient_id': pid}
    for i, label in enumerate(np.unique(np.concatenate([pred, gt]))):
        metrics[f"Dice_{label}"] = float(dice[i])
        metrics[f"HD95_{label}"] = float(hd95[i])
        metrics[f"Sens_{label}"] = float(sens[i])
        metrics[f"Spec_{label}"] = float(spec[i])

    return metrics


def evaluate_batch(pred_dir: str, gt_dir: str, patient_ids=None):
    """
    Evaluate all patients in directories. Returns:
      - df: pandas DataFrame of per-patient metrics
      - summary: dict of aggregated means across patients
    """
    files = [f for f in os.listdir(pred_dir) if f.startswith("seg_") and f.endswith(".nii.gz")]
    if patient_ids:
        files = [f for f in files if any(pid in f for pid in patient_ids)]

    records = []
    for fname in files:
        pid = fname.split('_')[1].split('.')[0]
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir, fname)
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for {pid}, skipping.")
            continue
        rec = evaluate_patient(pred_path, gt_path)
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    summary = df.drop(columns=["patient_id"]).mean().to_dict()
    return df, summary


def save_summary(summary: dict, out_path: str):
    """
    Save aggregated summary metrics to JSON.
    """
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

# For backward compatibility
evaluate_patient.save_metrics = staticmethod(
    lambda metrics, path: pd.DataFrame([metrics]).to_csv(path, index=False)
)

evaluate_batch.save_summary = staticmethod(save_summary)