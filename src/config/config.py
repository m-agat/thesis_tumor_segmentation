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

# Ensure repo root on path
current_dir = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import dataset.dataloaders as dataloaders  # noqa: E402

# ---------------------------------
# 1. Configuration File Loader
# ---------------------------------

def load_config(file_path=None):
    """
    Load JSON configuration from file and convert Windows paths to WSL/Linux if needed.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = file_path or os.path.join(base_dir, "config.json")
    with open(file_path, "r") as f:
        cfg = json.load(f)

    # Convert Windows paths if running under WSL
    if platform.system() == "Linux" and "microsoft" in platform.uname().release:
        for key in ["root_dir_local", "default_model_dir"]:
            if key in cfg:
                cfg[key] = convert_path(cfg[key])
    return cfg


def convert_path(path):
    """
    Convert Windows-style paths (e.g. C:\\\\path\\to\\file) to WSL/Linux format via wslpath.
    Returns unchanged if already Linux or URL.
    """
    if not path or path.startswith(("/", "https://")):
        return path
    if re.match(r"^[a-zA-Z]:\\", path):
        try:
            res = subprocess.run(["wslpath", "-u", path], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except subprocess.CalledProcessError:
            return path
    return path

# ---------------------------------
# 2. Command-line Argument Parser
# ---------------------------------

def _create_config_parser():
    """
    Internal: define arguments for configuration and return ArgumentParser.
    Uses add_help=False and parse_known_args so it won't error on unknown flags.
    """
    parser = argparse.ArgumentParser(add_help=False)
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
    parser.add_argument(
        "--method",
        type=str,
        choices=["simple", "perf", "tta", "ttd", "hybrid"],
        help="Which fusion strategy to run",
    )
    parser.add_argument(
        "--n_iter", type=int, default=10,
        help="Number of TTA/TTD samples (only used for tta, ttd, hybrid)"
    )
    parser.add_argument(
        "--patient_id", type=str, default=None,
        help="(Optional) restrict inference to this one patient ID"
    )
    return parser


def parse_args():
    """
    Parse only known arguments for config, ignoring others.
    Returns argparse Namespace.
    """
    parser = _create_config_parser()
    args, _unknown = parser.parse_known_args()
    return args

# ---------------------------------
# 3. Lazy Initialization of Config Values
# ---------------------------------

# Only executed at import time, but won’t error on unknown flags
args = parse_args()
raw_cfg = load_config()

# Project root (two levels up from src/config)
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Data directories
root_dir = args.data_path or raw_cfg.get(
    "root_dir", os.path.join(project_root, "data")
)
json_path = args.data_path or raw_cfg.get(
    "json_path", os.path.join(root_dir, "splits/final_training_splits.json")
)
train_folder = convert_path(os.path.join(root_dir, raw_cfg.get("train_subdir", "split/train")))
val_folder = convert_path(os.path.join(root_dir, raw_cfg.get("val_subdir", "split/val")))
test_folder = convert_path(os.path.join(root_dir, raw_cfg.get("test_subdir", "split/test")))
cross_val_folder = convert_path(os.path.join(root_dir, raw_cfg.get("cross_val_subdir", "split/cross_val_train")))

# Model paths
model_paths = {
    "swinunetr": os.path.join(
        args.swinunetr_path or raw_cfg.get("swinunetr_path", os.path.join(project_root, "src/models/saved_models/")),
        "best_swinunetr_model.pt",
    ),
    "segresnet": os.path.join(
        args.segresnet_path or raw_cfg.get("segresnet_path", os.path.join(project_root, "src/models/saved_models/")),
        "best_segresnet_model.pt",
    ),
    "attunet": os.path.join(
        args.attunet_path or raw_cfg.get("attunet_path", os.path.join(project_root, "src/models/saved_models/")),
        "best_attunet_model.pt",
    ),
    "vnet": os.path.join(
        args.vnet_path or raw_cfg.get("vnet_path", os.path.join(project_root, "src/models/saved_models/")),
        "best_vnet_model.pt",
    ),
}

# Output
output_dir = os.path.join(args.output_path, args.model_name)

# Device and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

roi = tuple(args.roi) if args.roi else tuple(raw_cfg.get("roi", [64, 64, 64]))
batch_size = args.batch_size or raw_cfg.get("batch_size", 1)
sw_batch_size = args.sw_batch_size or raw_cfg.get("sw_batch_size", 1)
infer_overlap = args.infer_overlap or raw_cfg.get("infer_overlap", 0.5)
max_epochs = raw_cfg.get("max_epochs", 100)
val_every = raw_cfg.get("val_every", 5)
num_folds = raw_cfg.get("num_folds", 5)

# Helper functions

def create_subset(data_loader, subset_size=10, shuffle=True):
    """Return a random subset DataLoader."""
    indices = random.sample(range(len(data_loader.dataset)), subset_size)
    subset = Subset(data_loader.dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def find_patient_by_id(patient_id, data_loader):
    """Restrict inference to a single patient in a DataLoader."""
    idx = next(
        (i for i, d in enumerate(data_loader.dataset)
         if re.search(r"BraTS2021_(\d+)", d["path"]) and
            re.search(r"BraTS2021_(\d+)", d["path"]).group(1) == patient_id
        ),
        None,
    )
    if idx is None:
        raise ValueError(f"Patient ID {patient_id} not found in dataset.")
    return DataLoader(Subset(data_loader.dataset, [idx]), batch_size=1, shuffle=False)

# Expose public API
__all__ = [
    "load_config", "convert_path", "parse_args", "raw_cfg", "args",
    "project_root", "root_dir", "json_path", "train_folder", "val_folder",
    "test_folder", "cross_val_folder", "model_paths", "output_dir",
    "device", "roi", "batch_size", "sw_batch_size", "infer_overlap",
    "max_epochs", "val_every", "num_folds", "create_subset",
    "find_patient_by_id",
]
