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

# Load configuration settings from JSON
def load_config(file_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = file_path or os.path.join(base_dir, "config.json")
    with open(file_path, "r") as f:
        config = json.load(f)
    # Convert paths in config to WSL format if necessary
    if platform.system() == "Linux" and "microsoft" in platform.uname().release:
        for key in ["root_dir_local", "default_model_dir"]:
            if key in config:
                config[key] = convert_path(config[key])
    return config

def convert_path(path):
    """Convert a Windows path to WSL format if needed, and return Azure or Linux paths as is."""
    if path is None:
        return None

    # If the path is already in a WSL/Linux format or is an Azure URL, return it as is
    if path.startswith("/") or path.startswith("https://"):
        return path

    # Convert Windows-style paths to WSL format
    if re.match(r"^[a-zA-Z]:\\", path):
        try:
            result = subprocess.run(['wslpath', '-u', path], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error converting path: {e}")
            return path  # Return original path if conversion fails

    return path

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument("--swinunetr_path", type=str, help="Path to the swinunetr model directory")
    parser.add_argument("--segresnet_path", type=str, help="Path to the segresnet model directory")
    parser.add_argument("--vnet_path", type=str, help="Path to the vnet model directory")
    parser.add_argument("--attunet_path", type=str, help="Path to the attunet model directory")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["swinunetr", "segresnet", "vnet", "attunet"],
        default="swinunetr",
        help="Model name to load",
    )
    parser.add_argument(
        "--output_path", type=str, default="./outputs", help="Path to save output files"
    )
    parser.add_argument(
        "--roi", type=int, nargs=3, default=[96, 96, 96], help="Region of interest size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for data loaders"
    )
    parser.add_argument(
        "--sw_batch_size", type=int, default=1, help="Sliding window batch size"
    )
    parser.add_argument(
        "--infer_overlap", type=float, default=0.5, help="Sliding window overlap"
    )
    parser.add_argument(
        "--subset_size", type=int, default=10, help="Size of subset for testing"
    )
    return parser.parse_args()

# Initialize configurations
args = parse_args()
config = load_config()

# Set paths based on args or config.json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Define root directory for data with priority: args -> config -> project_root default
root_dir = args.data_path or config.get("root_dir_local", "/home/magata/data/brats2021challenge")

# Define train, val, and test folders relative to root_dir
train_folder = convert_path(os.path.join(root_dir, config.get("train_subdir", "split/train")))
val_folder = convert_path(os.path.join(root_dir, config.get("val_subdir", "split/val")))
test_folder = convert_path(os.path.join(root_dir, config.get("test_subdir", "split/test")))

# Model paths
default_model_dir = convert_path(
    config.get("default_model_dir") or os.path.join(project_root, "results", "SwinUNetr")
)
model_file_path = os.path.join(
    args.model_path or config.get("default_model_dir", "/home/magata/results/SwinUNetr"),
    f"{args.model_name}_model.pt",
)
model_paths = {
    "swinunetr": os.path.join(
        args.swinunetr_path or "/home/magata/results/SwinUNetr",
        "swinunetr_model.pt"
        ),
    "segresnet": os.path.join(
        args.segresnet_path or "/home/magata/results/SegResNet",
        "segresnet_model.pt"
        ),
    "attunet": os.path.join(
        args.attunet_path or "/home/magata/results/AttentionUNet",
        "attunet_model.pt"
        ),
    "vnet": os.path.join(
        args.vnet_path or "/home/magata/results/VNet",
        "vnet_model.pt"
        )
    }  

output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)

print("Project root:", project_root)
print("Root directory for data:", root_dir)
print("Train folder:", train_folder)
print("Validation folder:", val_folder)
print("Test folder:", test_folder)
print("Output directory:", output_dir)
print("Default model path:", model_file_path)
print("All model paths:", model_paths)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Parameters
roi = tuple(args.roi if args.roi else config.get("roi", [64, 64, 64]))
batch_size = args.batch_size or config.get("batch_size", 1)
sw_batch_size = args.sw_batch_size or config.get("sw_batch_size", 1)
infer_overlap = args.infer_overlap or config.get("infer_overlap", 0.6)

# Initialize data loaders
train_loader, val_loader = dataloaders.get_loaders(batch_size, train_folder, val_folder, roi)
test_loader = dataloaders.load_test_data(test_folder)
print("Data loaders loaded")

# Helper function to create a subset of a DataLoader
def create_subset(data_loader, subset_size=10, shuffle=True):
    indices = random.sample(range(len(data_loader.dataset)), subset_size)
    subset = Subset(data_loader.dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle) 

# Helper function to find and load a specific patient by ID
def find_patient_by_id(patient_id, data_loader):
    index_to_include = next(
        (idx for idx, data in enumerate(data_loader.dataset) if data["path"] == patient_id),
        None,
    )
    if index_to_include is None:
        raise ValueError(f"Patient ID {patient_id} not found in the test dataset.")
    return DataLoader(
        Subset(data_loader.dataset, [index_to_include]), batch_size=1, shuffle=False
    )

# Initialize subsets
train_loader_subset = create_subset(train_loader, subset_size=args.subset_size)
val_loader_subset = create_subset(val_loader, subset_size=args.subset_size)
test_loader_subset = create_subset(test_loader, subset_size=args.subset_size)

# Initialize specific patient loader for testing
patient_id_to_find = "BraTS2021_01339"  # Modify as needed
test_loader_patient = find_patient_by_id(patient_id_to_find, test_loader)

# Set attributes to be accessible in the main code
model_name = args.model_name
print(
    f"Configuration paths:\n Train: {train_folder}\n Val: {val_folder}\n Test: {test_folder}"
)
print(f"Model file: {model_file_path}\n Output directory: {output_dir}")
print(f"Device: {device}\n ROI: {roi}\n Batch Size: {batch_size}\n Sliding Window Batch Size: {sw_batch_size}\n Inference Overlap: {infer_overlap}\n")
