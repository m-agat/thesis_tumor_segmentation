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


class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class EncoderBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock3D(in_c, out_c)
        self.pool = nn.MaxPool3d((2, 2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class DecoderBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock3D(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=5):  # Change to 5 output channels
        super().__init__()
        self.e1 = EncoderBlock3D(in_channels, 64)
        self.e2 = EncoderBlock3D(64, 128)
        self.e3 = EncoderBlock3D(128, 256)
        self.e4 = EncoderBlock3D(256, 512)
        self.b = ConvBlock3D(512, 1024)
        self.d1 = DecoderBlock3D(1024, 512)
        self.d2 = DecoderBlock3D(512, 256)
        self.d3 = DecoderBlock3D(256, 128)
        self.d4 = DecoderBlock3D(128, 64)
        self.outputs = nn.Conv3d(64, out_channels, kernel_size=1, padding=0)  # Output channels = 5

    def forward(self, inputs):
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        b = self.b(p4)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        outputs = self.outputs(d4)
        return outputs


    

