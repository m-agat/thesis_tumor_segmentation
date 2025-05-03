#!/usr/bin/env python3
import os
import re
from pathlib import Path

# 1) directory that contains the model sub-folders
ROOT = Path("./segmentations/")

# 2) folders you want to process
MODEL_FOLDERS = ["hybrid", "tta", "ttd"]

# --- helper to rename a single file ------------------------------------------
def rename_seg(file_path: Path, model_name: str):
    """
    Rename  seg_VIGO_01.nii.gz  ->  tta_VIGO_01_pred_seg.nii.gz  (example)
    """
    m = re.match(r"seg_(.+)\.nii\.gz$", file_path.name)
    if not m:
        return  # not a segmentation file we care about

    patient_id = m.group(1)
    new_name = f"{model_name}_{patient_id}_pred_seg.nii.gz"
    new_path = file_path.with_name(new_name)

    print(f"↪  {file_path.relative_to(ROOT.parent)}  →  {new_path.name}")
    file_path.rename(new_path)

# --- main loop ----------------------------------------------------------------
for model_name in MODEL_FOLDERS:
    folder = ROOT / model_name
    if not folder.is_dir():
        print(f"⚠  {folder} does not exist — skipping")
        continue

    for item in folder.iterdir():
        if item.is_file():
            rename_seg(item, model_name)

print("✅  renaming done")
