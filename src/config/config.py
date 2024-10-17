import sys
sys.path.append('../')
import os
import dataset.dataloaders as dataloaders
import torch 
from torch.utils.data import Subset
import random 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to the dataset')
    return parser.parse_args()

args = parse_args()

if args.data_path:
    # Azure mode (mounted Blob Storage)
    root_dir = args.data_path
    train_folder = os.path.join(root_dir, "train")
    val_folder = os.path.join(root_dir, "val")
    print(f"Running on Azure. Dataset is mounted at {root_dir}")
else:
    # Local mode
    root_dir = "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge"
    train_folder = os.path.join(root_dir, "split/train")
    val_folder = os.path.join(root_dir, "split/val")
    print(f"Running locally. Using local dataset at {root_dir}")

# Show the paths to the folders with the dataset
print(f"Train folder: {train_folder}")
print(f"Val folder: {val_folder}")

# GPU config
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"Using {device}")

# Global configuration parameters 
roi = (96, 96, 96)
batch_size = 1
sw_batch_size = 2
infer_overlap = 0.5
max_epochs = 1 
val_every = 10

# Data loaders for binary segmentation
train_loader, val_loader = dataloaders.get_loaders(batch_size, train_folder, val_folder, roi)