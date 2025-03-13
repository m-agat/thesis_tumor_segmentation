import os
from monai import data
from dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)
import json
import numpy as np
from torch.utils.data import Subset


def get_subset(dataset, fraction, seed):
    np.random.seed(seed)
    subset_size = int(len(dataset) * fraction)
    indices = np.random.choice(
        len(dataset), subset_size, replace=False
    )  # Random subset
    return Subset(dataset, indices)


def load_folds_data(json_path, basedir, fold=None, use_final_split=False):
    """
    Load training and validation data for a specific fold from the JSON file.

    Can be used for both CV and final training.

    Args:
        json_path (str): Path to the JSON file containing data splits.
        basedir (str): Base directory to prepend to the file paths.
        fold (int): Fold number to select validation data.

    Returns:
        tuple: A tuple containing training data and validation data lists.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    training_files = []
    validation_files = []

    if use_final_split:
        for entry in data["training"]:
            entry["image"] = [os.path.join(basedir, path) for path in entry["image"]]
            entry["label"] = os.path.join(basedir, entry["label"])
            training_files.append(entry)

        for entry in data["validation"]:
            entry["image"] = [os.path.join(basedir, path) for path in entry["image"]]
            entry["label"] = os.path.join(basedir, entry["label"])
            validation_files.append(entry)
    else:
        for entry in data["training"]:
            if entry.get("fold") == fold:
                entry["image"] = [
                    os.path.join(basedir, path) for path in entry["image"]
                ]
                entry["label"] = os.path.join(basedir, entry["label"])
                training_files.append(entry)

        for entry in data["validation"]:
            if entry.get("fold") == fold:
                entry["image"] = [
                    os.path.join(basedir, path) for path in entry["image"]
                ]
                entry["label"] = os.path.join(basedir, entry["label"])
                validation_files.append(entry)

    return training_files, validation_files


def get_loaders(
    batch_size, json_path, basedir, fold=None, roi=None, use_final_split=False
):
    """
    Create data loaders for the second stage, focusing on multi-class segmentation.
    """
    # Load train and validation files
    train_files, val_files = load_folds_data(
        json_path=json_path, basedir=basedir, fold=fold, use_final_split=use_final_split
    )

    # Get the transforms for multi-class segmentation
    train_transform = get_train_transforms(roi)
    val_transform = get_val_transforms()

    # Create datasets with patches
    train_ds = data.Dataset(data=train_files, transform=train_transform)
    val_ds = data.Dataset(data=val_files, transform=val_transform)

    # Create data loaders
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader


def load_test_data(json_path, basedir, batch_size=1):
    """
    Load all test files from a specified test folder and apply transformations.
    """
    # List all cases in the test folder
    with open(json_path, "r") as f:
        json_data = json.load(f)

    test_files = []
    for entry in json_data["test"]:
        entry["image"] = [os.path.join(basedir, path) for path in entry["image"]]
        entry["label"] = os.path.join(basedir, entry["label"])
        entry["path"] = entry["label"]
        test_files.append(entry)

    # Get the test transforms
    test_transform = get_test_transforms()

    # Create test dataset
    test_ds = data.Dataset(data=test_files, transform=test_transform)

    # Create test DataLoader
    test_loader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return test_loader
