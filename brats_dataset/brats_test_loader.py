import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainTumorValidationDatasetNifti(Dataset):
    def __init__(self, image_dir, transform=None, volume_depth=16):
        self.image_dir = image_dir
        self.transform = transform
        self.cases = os.listdir(image_dir)
        self.volume_depth = volume_depth

    def __len__(self):
        return len(self.cases) * self._get_num_slices_per_volume()

    def _get_num_slices_per_volume(self):
        example_case = os.path.join(self.image_dir, self.cases[0])
        example_nii = nib.load(os.path.join(example_case, f"{self.cases[0]}-t1c.nii.gz"))
        return example_nii.shape[2] // self.volume_depth

    def __getitem__(self, idx):
        case_idx = idx // self._get_num_slices_per_volume()
        slice_idx = (idx % self._get_num_slices_per_volume()) * self.volume_depth
        
        case = self.cases[case_idx]
        case_dir = os.path.join(self.image_dir, case)
        
        t1c = nib.load(os.path.join(case_dir, f"{case}-t1c.nii.gz")).get_fdata()
        t1n = nib.load(os.path.join(case_dir, f"{case}-t1n.nii.gz")).get_fdata()
        t2f = nib.load(os.path.join(case_dir, f"{case}-t2f.nii.gz")).get_fdata()
        t2w = nib.load(os.path.join(case_dir, f"{case}-t2w.nii.gz")).get_fdata()

        # Stack modalities to create a multi-channel input volume
        image = np.stack([t1c, t1n, t2f, t2w], axis=0)

        # Select the volume of depth `self.volume_depth`
        image = image[:, slice_idx:slice_idx+self.volume_depth, :, :]

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)

        # Convert to torch tensor
        image = torch.from_numpy(image).float()

        return {"image": image, "case": case, "start_slice": slice_idx}

