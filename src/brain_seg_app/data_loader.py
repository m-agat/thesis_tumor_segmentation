import nibabel as nib
import numpy as np
import torch
from monai import data
from config_web import BATCH_SIZE
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.transforms import get_test_transforms


def create_test_loader(mri_scans):
    """
    Constructs a MONAI DataLoader from a list of MRI scan file paths.
    Here, mri_scans is a list containing file paths for each modality.
    """
    print(mri_scans)
    first_img = nib.load(mri_scans[0])
    shape = first_img.get_fdata().shape      
    # Create a dummy label with the same spatial dimensions as shape var
    dummy_label_array = np.zeros(shape, dtype=np.uint8)
    dummy_label_path = os.path.join(os.getcwd(), "dummy_label.nii.gz")
    dummy_label_nii = nib.Nifti1Image(dummy_label_array, affine=first_img.affine)
    nib.save(dummy_label_nii, dummy_label_path)

    entry = {
        "image": mri_scans,       # list of file paths, one per modality
        "label": dummy_label_path,     # dummy label
        "path": mri_scans[0]      # use the first modality's path as reference (e.g., for patient ID)
    }
    test_files = [entry]

    test_transform = get_test_transforms()

    test_ds = data.Dataset(data=test_files, transform=test_transform)
    test_loader = data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    return test_loader
