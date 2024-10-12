import os
import dataset.dataloaders_binary as dataloaders_binary
import torch 

# Set the root directory 
root_dir = "/home/agata/Desktop/thesis_tumor_segmentation/"

# Define your parameters
data_dir = os.path.join(root_dir, "data/brats2021challenge")
train_folder = "/split/train"
val_folder = "split/val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_roi = (128, 128, 128)
local_roi = (64, 64, 64)
batch_size = 1
sw_batch_size = 2
infer_overlap = 0.5
max_epochs = 100
val_every = 10
train_loader, val_loader = dataloaders_binary.get_loader(batch_size, data_dir, train_folder, val_folder, global_roi, local_roi)