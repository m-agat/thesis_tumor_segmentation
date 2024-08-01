import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainTumorDatasetNifti(Dataset):
    def __init__(self, image_dir, transform=None, cases_subset=None, volume_depth=16):
        self.image_dir = image_dir
        self.transform = transform
        self.cases = os.listdir(image_dir)
        self.volume_depth = volume_depth

        if cases_subset:
            self.cases = [case for case in self.cases if case in cases_subset]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.image_dir, case)
        
        t1c = nib.load(os.path.join(case_dir, f"{case}-t1c.nii.gz")).get_fdata()
        t1n = nib.load(os.path.join(case_dir, f"{case}-t1n.nii.gz")).get_fdata()
        t2f = nib.load(os.path.join(case_dir, f"{case}-t2f.nii.gz")).get_fdata()
        t2w = nib.load(os.path.join(case_dir, f"{case}-t2w.nii.gz")).get_fdata()
        seg = nib.load(os.path.join(case_dir, f"{case}-seg.nii.gz")).get_fdata()

        # Stack modalities to create a multi-channel input volume
        image = np.stack([t1c, t1n, t2f, t2w], axis=0)

        # Ensure volume depth does not exceed available depth
        max_depth = image.shape[1] - self.volume_depth
        if max_depth < 0:
            raise ValueError(f"Volume depth {self.volume_depth} exceeds the available depth {image.shape[1]} in case {case}")

        # Select a random depth slice to create a volume of depth `self.volume_depth`
        start_slice = np.random.randint(0, max_depth + 1)
        image = image[:, start_slice:start_slice+self.volume_depth, :, :]
        mask = seg[start_slice:start_slice+self.volume_depth, :, :]

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        # Ensure the mask has the correct shape (B, D, H, W)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0)  # Add a channel dimension

        return {"image": image, "label": mask}
