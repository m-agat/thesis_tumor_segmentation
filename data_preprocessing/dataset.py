import os
import torch
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

'''
Adapted from https://github.com/RosalindFok/3D-BrainTumorSegmentat4MIA/blob/main/read_nii_utils.py
'''

class BraTSDataset(Dataset):
    def __init__(self, subjects_list: list, transform=None, target_shape=(160, 192, 128), load_segmentation=True):
        """
        Dataset loader for BraTS 2024 NIfTI data.

        :param subjects_list: List of paths to the subjects' directories.
        :param transform: Optional transform to apply on data.
        :param target_shape: Shape to crop or pad the input data to.
        :param load_segmentation: Whether to load segmentation masks (use False for validation set).
        """
        super().__init__()
        self.nii_path_list = [[os.path.join(subject, file_name) for file_name in os.listdir(subject)]
                              for subject in subjects_list]
        self.transform = transform
        self.target_shape = target_shape
        self.load_segmentation = load_segmentation

    def __getitem__(self, idx):
        all_files = self.nii_path_list[idx]

        # Load multimodal scans
        t1c = min_max_normalize(crop_nparray(read_nii_data(find_first_match('t1c.', all_files)), self.target_shape))
        t1n = min_max_normalize(crop_nparray(read_nii_data(find_first_match('t1n.', all_files)), self.target_shape))
        t2f = min_max_normalize(crop_nparray(read_nii_data(find_first_match('t2f.', all_files)), self.target_shape))
        t2w = min_max_normalize(crop_nparray(read_nii_data(find_first_match('t2w.', all_files)), self.target_shape))

        # Stack the modalities into a single tensor
        input_tensor = torch.Tensor(np.stack([t1c, t1n, t2f, t2w]))

        # Only load segmentation if required (e.g., for training or testing, not validation)
        if self.load_segmentation:
            seg = crop_nparray(read_nii_data(find_first_match('seg.', all_files)), self.target_shape)
            seg_tensor = torch.Tensor(seg)
        else:
            seg_tensor = None  # Placeholder for validation

        # Apply any transforms (data augmentation, etc.)
        if self.transform:
            input_tensor, seg_tensor = self.transform(input_tensor, seg_tensor)

        return input_tensor, seg_tensor if seg_tensor is not None else torch.zeros(input_tensor.shape[1:])

    def __len__(self):
        return len(self.nii_path_list)

def read_nii_data(file_path):
    """Read a NIfTI file and return its data."""
    return nib.load(file_path).get_fdata()

def find_first_match(substr, str_list):
    """Find the first file in a list containing the substring."""
    return [x for x in str_list if substr in x][0]

def crop_nparray(original_matrix, target_shape):
    """Crop the center part of a numpy array to the target shape."""
    start_indices = [(dim - target_dim) // 2 for dim, target_dim in zip(original_matrix.shape, target_shape)]
    end_indices = [start + target_dim for start, target_dim in zip(start_indices, target_shape)]
    cropped_matrix = original_matrix[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
    return cropped_matrix

def min_max_normalize(data):
    """Normalize data to [0, 1] range using min-max normalization."""
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max != data_min:
        return (data - data_min) / (data_max - data_min)
    return np.ones_like(data)  # Avoid division by zero

