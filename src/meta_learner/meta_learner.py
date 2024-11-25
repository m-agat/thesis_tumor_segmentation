import os
import sys
import pandas as pd
import numpy as np
import ast
sys.path.append("../")

from load_outputs import save_file_paths
import nibabel as nib 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config.config as config
from train.train_helpers import train_epoch
from load_outputs import MetaLearnerDataset
from torch.utils.data import Dataset, DataLoader
from monai.losses import GeneralizedDiceFocalLoss


base_path = "./outputs/"  # The path where the model outputs (segmentation, probability maps, and logits) are stored
models = ["swinunetr", "segresnet", "vnet", "attunet"]  
gt_path = "/home/magata/data/brats2021challenge/split/val/" 
data_df = pd.read_csv("./outputs/model_file_paths_testing.csv")

class EnsembleCNN(nn.Module):
    def __init__(self, input_channels=12, output_channels=3):
        super(EnsembleCNN, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)

        # No hardcoded fc_input_dim; we will compute it dynamically
        self.fc = nn.Linear(512, output_channels)  # Placeholder for final output

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 4, 3, H, W, D].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_patches, output_channels].
        """
        # Merge models and tissue types into channels
        batch_size, M, C, H, W, D = x.shape
        x = x.view(batch_size, M * C, H, W, D)  # [batch_size, 12, H, W, D]

        # Apply 3D convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)  # Flatten dynamically
        print(f"Flattened tensor shape before fc: {x.shape}")  # Debugging

        # Update fc dynamically
        fc_input_dim = x.size(1)
        if not hasattr(self.fc, "initialized"):
            self.fc = nn.Linear(fc_input_dim, 3).to(x.device)  # Dynamically initialize fc
            self.fc.initialized = True

        x = self.fc(x)  # Apply fully connected layer
        return x
    
# data_df = pd.read_csv("./outputs/model_file_paths_with_gt.csv") 
data_df = pd.read_csv("./outputs/model_file_paths_testing.csv") 

dataset = MetaLearnerDataset(data_df, models, patch_size=(64, 64, 64), stride=(32, 32, 32))

batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Just for checking the shapes
for idx, batch_data in enumerate(dataloader):
    print(batch_data["image"].shape, batch_data["label"].shape) 
    break

model = EnsembleCNN().to(config.device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_func = GeneralizedDiceFocalLoss(
    include_background=True, to_onehot_y=False, sigmoid=True, w_type="square"
)
max_epochs = 1
for epoch in range(max_epochs):
    train_loss = train_epoch(model, dataloader, optimizer, epoch, loss_func=loss_func)
    print(f"Epoch {epoch}, Loss: {train_loss:.4f}")
