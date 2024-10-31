import sys
sys.path.append("../")
import os
import dataset.dataloaders as dataloaders
import torch
import argparse
import random
from torch.utils.data import Subset, DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    return parser.parse_args()


args = parse_args()

if args.data_path:
    # Azure mode (mounted Blob Storage)
    root_dir = args.data_path
    train_folder = os.path.join(root_dir, "train")
    val_folder = os.path.join(root_dir, "val")
    test_folder = os.path.join(root_dir, "test")
    print(f"Running on Azure. Dataset is mounted at {root_dir}")
else:
    # Local mode
    root_dir = "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge"
    train_folder = os.path.join(root_dir, "split/train")
    val_folder = os.path.join(root_dir, "split/val")
    test_folder = os.path.join(root_dir, "split/test")
    print(f"Running locally. Using local dataset at {root_dir}")

if args.model_path:
    model_file_path = os.path.join(args.model_path, "swinunetr_model.pt")
else:
    model_file_path = "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/swinunetr_model.pt"

# Show the paths to the folders with the dataset
print(f"Train folder: {train_folder}")
print(f"Val folder: {val_folder}")
print(f"Test folder: {test_folder}")

print(f"Model file path: {model_file_path}")

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
max_epochs = 300
val_every = 5

# Data loaders
train_loader, val_loader = dataloaders.get_loaders(
    batch_size, train_folder, val_folder, roi
)
test_loader = dataloaders.load_test_data(test_folder)

# Creating subset of data for testing purposes
# subset_size = 10

# # Create random indices for the subset
# train_indices = random.sample(range(len(train_loader.dataset)), subset_size)
# val_indices = random.sample(range(len(val_loader.dataset)), subset_size)
# test_indices = random.sample(range(len(test_loader.dataset)), subset_size)

# # Create subset data loaders
# train_subset = Subset(train_loader.dataset, train_indices)
# val_subset = Subset(val_loader.dataset, val_indices)
# test_subset = Subset(test_loader.dataset, test_indices)

# # New data loaders with the smaller subsets
# train_loader_subset = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
# val_loader_subset = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
# test_loader_subset = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

patient_id_to_find = "BraTS2021_01532"
index_to_include = None

for idx, data in enumerate(test_loader.dataset):
    if data['path'] == patient_id_to_find:
        index_to_include = idx
        break

if index_to_include is None:
    raise ValueError(f"Patient ID {patient_id_to_find} not found in the test dataset.")

test_subset = Subset(test_loader.dataset, [index_to_include])
test_loader_subset = DataLoader(test_subset, batch_size=batch_size, shuffle=False)