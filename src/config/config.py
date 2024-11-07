import sys
import os
import json
import argparse
import random
import torch
from torch.utils.data import DataLoader, Subset

sys.path.append("../")
import dataset.dataloaders as dataloaders

# Load configuration settings from JSON
def load_config(file_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = file_path or os.path.join(base_dir, "config.json")
    with open(file_path, "r") as f:
        return json.load(f)

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, help="Path to the model directory")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["swinunetr", "segresnet", "vnet", "attentionunet"],
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
        "--sw_batch_size", type=int, default=2, help="Sliding window batch size"
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
root_dir = args.data_path or config.get(
    "root_dir_local",
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge",
)
train_folder = os.path.join(root_dir, "train") if args.data_path else os.path.join(root_dir, "split/train")
val_folder = os.path.join(root_dir, "val") if args.data_path else os.path.join(root_dir, "split/val")
test_folder = os.path.join(root_dir, "test") if args.data_path else os.path.join(root_dir, "split/test")

# Model and output paths
model_file_path = os.path.join(
    args.model_path or config.get("default_model_dir", "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr"),
    f"{args.model_name}_model.pt",
)
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Parameters
roi = tuple(args.roi if args.roi else config.get("roi", [96, 96, 96]))
batch_size = args.batch_size or config.get("batch_size", 1)
sw_batch_size = args.sw_batch_size or config.get("sw_batch_size", 2)
infer_overlap = args.infer_overlap or config.get("infer_overlap", 0.5)

# Initialize data loaders
train_loader, val_loader = dataloaders.get_loaders(batch_size, train_folder, val_folder, roi)
test_loader = dataloaders.load_test_data(test_folder)

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
patient_id_to_find = "BraTS2021_01532"  # Modify as needed
test_loader_patient = find_patient_by_id(patient_id_to_find, test_loader)

# Set attributes to be accessible in the main code
model_name = args.model_name
print(
    f"Configuration paths:\n Train: {train_folder}\n Val: {val_folder}\n Test: {test_folder}"
)
print(f"Model file: {model_file_path}\n Output directory: {output_dir}")
print(f"Device: {device}\n ROI: {roi}\n Batch Size: {batch_size}\n Sliding Window Batch Size: {sw_batch_size}\n Inference Overlap: {infer_overlap}\n")
