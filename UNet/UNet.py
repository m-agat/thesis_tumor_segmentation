import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import nibabel as nib
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    

class BuildUNet(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = EncoderBlock(4, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)
        """ Bottleneck """
        self.b = ConvBlock(512, 1024)
        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs
    

class BrainTumorDatasetNifti(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.cases = os.listdir(image_dir)

    def __len__(self):
        return len(self.cases) * self._get_num_slices()

    def _get_num_slices(self):
        # Assumes all cases have the same number of slices
        example_case = os.path.join(self.image_dir, self.cases[0])
        example_nii = nib.load(os.path.join(example_case, os.listdir(example_case)[0]))
        return example_nii.shape[2]

    def __getitem__(self, idx):
        case_idx = idx // self._get_num_slices()
        slice_idx = idx % self._get_num_slices()
        
        case = self.cases[case_idx]
        case_dir = os.path.join(self.image_dir, case)
        
        t1c = nib.load(os.path.join(case_dir, f"{case}-t1c.nii.gz")).get_fdata()[:, :, slice_idx]
        t1n = nib.load(os.path.join(case_dir, f"{case}-t1n.nii.gz")).get_fdata()[:, :, slice_idx]
        t2f = nib.load(os.path.join(case_dir, f"{case}-t2f.nii.gz")).get_fdata()[:, :, slice_idx]
        t2w = nib.load(os.path.join(case_dir, f"{case}-t2w.nii.gz")).get_fdata()[:, :, slice_idx]
        seg = nib.load(os.path.join(case_dir, f"{case}-seg.nii.gz")).get_fdata()[:, :, slice_idx]
        
        # Stack the modalities to create a multi-channel input
        image = np.stack([t1c, t1n, t2f, t2w], axis=0)
        mask = seg[np.newaxis, :, :]  # Add channel dimension to mask
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()

        return {"image": image, "label": mask}



