import os
import nibabel as nib
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 4 
OOD_DATA_PATH = "/home/magata/data/braintumor_data"
WTS = {"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}

def load_nifti(filepath):
    """
    Load a NIfTI file and return a tuple (data, affine, header).
    """
    img = nib.load(filepath)
    return img.get_fdata(), img.affine, img.header


def to_one_hot(volume: np.ndarray, n_classes: int = N_CLASSES):
    """
    Convert a label volume (H,W,D) into a list of boolean tensors,
    one per class, on the chosen device.
    """
    return [
        torch.from_numpy((volume == c).astype(np.uint8)).to(DEVICE)
        for c in range(n_classes)
    ]


def get_ood_cases():
    """Get list of out-of-distribution case folders."""
    if not os.path.exists(OOD_DATA_PATH):
        raise FileNotFoundError(f"OOD data directory not found at {OOD_DATA_PATH}")
    return [os.path.join(OOD_DATA_PATH, f, "original") for f in os.listdir(OOD_DATA_PATH) 
            if os.path.isdir(os.path.join(OOD_DATA_PATH, f))]