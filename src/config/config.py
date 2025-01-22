import sys
import os
import json
import argparse
import random
import platform
import re
import subprocess
import torch
from torch.utils.data import DataLoader, Subset
sys.path.append("../")
import dataset.dataloaders as dataloaders
import dataset.dataloaders_crossval as dataloaders_cv
import dataset.transforms as transforms 
import numpy as np

# ---------------------------------
# 1. Configuration File Loaders
# ---------------------------------

def load_config(file_path=None):
    """
    Load configuration from JSON file.
    Convert paths to WSL format if needed.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = file_path or os.path.join(base_dir, "config.json")
    
    with open(file_path, "r") as f:
        config = json.load(f)
    
    # Convert paths if on WSL
    if platform.system() == "Linux" and "microsoft" in platform.uname().release:
        for key in ["root_dir_local", "default_model_dir"]:
            if key in config:
                config[key] = convert_path(config[key])
    
    return config

def convert_path(path):
    """
    Convert Windows paths to WSL/Linux format.
    """
    if path is None or path.startswith(("/", "https://")):
        return path  # Return as-is if already in Linux or URL format

    if re.match(r"^[a-zA-Z]:\\", path):  # Convert Windows-style paths
        try:
            result = subprocess.run(['wslpath', '-u', path], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error converting path: {e}")
            return path  # Return original path if conversion fails
    return path

# ---------------------------------
# 2. Command-Line Arguments Parsing
# ---------------------------------
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument("--swinunetr_path", type=str, help="Path to the swinunetr model directory")
    parser.add_argument("--segresnet_path", type=str, help="Path to the segresnet model directory")
    parser.add_argument("--vnet_path", type=str, help="Path to the vnet model directory")
    parser.add_argument("--attunet_path", type=str, help="Path to the attunet model directory")
    parser.add_argument("--model_name", type=str, choices=["swinunetr", "segresnet", "vnet", "attunet"],
                        default="swinunetr", help="Model name to load")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Path to save output files")
    parser.add_argument("--roi", type=int, nargs=3, default=[96, 96, 96], help="Region of interest size")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for data loaders")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="Sliding window batch size")
    parser.add_argument("--infer_overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--subset_size", type=int, default=10, help="Size of subset for testing")
    return parser.parse_args()

# ---------------------------------
# 3. Initialization Logic
# ---------------------------------
# Load arguments and config
args = parse_args()
config = load_config()

# Root directories and paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
root_dir = args.data_path or config.get("root_dir", "/home/magata/data/brats2021challenge")
json_path = args.data_path or config.get("json_path", "/home/magata/data/brats2021challenge/splits/data_splits.json")

train_folder = convert_path(os.path.join(root_dir, config.get("train_subdir", "split/train")))
val_folder = convert_path(os.path.join(root_dir, config.get("val_subdir", "split/val")))
test_folder = convert_path(os.path.join(root_dir, config.get("test_subdir", "split/test")))
cross_val_folder = convert_path(os.path.join(root_dir, config.get("cross_val_subdir", "split/cross_val_train")))

# Model directories
model_name = args.model_name
default_model_dir = convert_path(config.get("default_model_dir") or os.path.join(project_root, "results", "SwinUNetr"))
model_file_path = os.path.join(args.model_path or config.get("default_model_dir", "/home/magata/results/SwinUNetr"),
                               f"{args.model_name}_model.pt")

model_paths = {
    "swinunetr": os.path.join(args.swinunetr_path or "/home/magata/results/SwinUNetr", "swinunetr_model.pt"),
    "segresnet": os.path.join(args.segresnet_path or "/home/magata/results/SegResNet", "segresnet_model.pt"),
    "attunet": os.path.join(args.attunet_path or "/home/magata/results/AttentionUNet", "attunet_model.pt"),
    "vnet": os.path.join(args.vnet_path or "/home/magata/results/VNet", "vnet_model.pt"),
}

# Output directory
output_dir = os.path.join(args.output_path, f"{model_name}")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------
# 4. Display Configuration
# ---------------------------------
def print_config_summary():
    print("\n")
    print("------ Config Summary ------")
    print("--------------------------------------")
    print(f"Project root: {project_root}")
    print(f"Root directory for data: {root_dir}")
    print(f"Model path: {model_file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"ROI: {roi}")
    print(f"Batch Size: {batch_size}")
    print(f"Sliding Window Batch Size: {sw_batch_size}")
    print(f"Inference Overlap: {infer_overlap}")
    print("--------------------------------------")
    print("\n")

# ---------------------------------
# 5. Device and Parameters
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

roi = tuple(args.roi if args.roi else config.get("roi", [64, 64, 64]))
batch_size = args.batch_size or config.get("batch_size", 1)
sw_batch_size = args.sw_batch_size or config.get("sw_batch_size", 1)
infer_overlap = args.infer_overlap or config.get("infer_overlap", 0.5)
max_epochs = config.get("max_epochs", 100)
val_every = config.get("val_every", 5)
num_folds = config.get("num_folds", 5)

# ---------------------------------
# 6. Data Loaders
# ---------------------------------
# Initialize data loaders
test_loader = dataloaders.load_test_data(json_path, root_dir)

# ---------------------------------
# 7. Helper Functions
# ---------------------------------
def create_subset(data_loader, subset_size=10, shuffle=True):
    indices = random.sample(range(len(data_loader.dataset)), subset_size)
    subset = Subset(data_loader.dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

def find_patient_by_id(patient_id, data_loader):
    index_to_include = next((idx for idx, data in enumerate(data_loader.dataset) if data["path"] == patient_id), None)
    if index_to_include is None:
        raise ValueError(f"Patient ID {patient_id} not found in the test dataset.")
    return DataLoader(Subset(data_loader.dataset, [index_to_include]), batch_size=1, shuffle=False)

# ---------------------------------
# 8. Final Configuration Summary
# ---------------------------------
print_config_summary()
