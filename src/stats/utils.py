import numpy as np
import os 

def format_patient_id(raw: str) -> str:
    """
    Turn something like 'case_00042' or 'brats_42' into '00042'.
    """
    return raw.split('_')[-1].zfill(5)

def format_patient_id_csv(x: int | str) -> str:
    """
    Zero‐pad an integer or numeric string into a 5-digit patient ID.
    """
    return str(int(x)).zfill(5)

def dice_column(region_label: str) -> str:
    """
    Given a region presence label like 'NCR present',
    return the corresponding Dice column: 'Dice NCR'.
    """
    return f"Dice {region_label.split()[0]}"

def compute_absence_performance(df, region_label: str):
    """
    For a merged DataFrame that has both predictions and
    a column e.g. 'NCR present'==0, compute the mean Dice
    on those absent cases and the count of cases.
    """
    absent = df[df[region_label] == 0]
    col = dice_column(region_label)
    mean = np.nanmean(absent[col])
    return mean, len(absent)

def is_ensemble_model(name: str) -> bool:
    """
    Detect common ensemble prefixes.
    """
    tags = ["Simple-Avg", "Performance-Weighted", "TTD", "Hybrid", "TTA"]
    return any(t in name for t in tags)

def format_relationship(g1: str, direction: str, g2: str) -> str:
    return f"{g1} {direction} {g2}"

def ensure_dir(path: str):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)