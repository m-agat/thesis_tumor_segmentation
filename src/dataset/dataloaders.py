import os
from monai import data
from dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)


def read_data_from_folders(train_folder, val_folder):
    """
    Read training and validation data from specific folders.
    """

    def load_cases_from_folder(folder):
        cases = [f for f in os.listdir(folder) if f.startswith("BraTS2021")]
        files = []
        for case in cases:
            files.append(
                {
                    "image": [
                        os.path.join(folder, case, f"{case}_flair.nii.gz"),
                        os.path.join(folder, case, f"{case}_t1ce.nii.gz"),
                        os.path.join(folder, case, f"{case}_t1.nii.gz"),
                        os.path.join(folder, case, f"{case}_t2.nii.gz"),
                    ],
                    "label": os.path.join(folder, case, f"{case}_seg.nii.gz"),
                    "path": case,
                }
            )
        return files

    # Load train and validation files
    train_files = load_cases_from_folder(train_folder)
    val_files = load_cases_from_folder(val_folder)

    return train_files, val_files


def get_loaders(batch_size, train_folder, val_folder, roi):
    """
    Create data loaders for the second stage, focusing on multi-class segmentation.
    """
    # Load train and validation files
    train_files, val_files = read_data_from_folders(train_folder, val_folder)

    # Get the global and local transforms for multi-class segmentation
    train_transform = get_train_transforms(roi)
    val_transform = get_val_transforms()

    # Create datasets with global and local patches
    mc_train_ds = data.Dataset(data=train_files, transform=train_transform)
    val_ds = data.Dataset(data=val_files, transform=val_transform)

    # Create data loaders
    local_train_loader = data.DataLoader(
        mc_train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    return local_train_loader, val_loader


def load_test_data(test_folder, batch_size=1):
    """
    Load all test files from a specified test folder and apply transformations.
    """
    # List all cases in the test folder
    test_cases = [f for f in os.listdir(test_folder) if f.startswith("BraTS2021")]

    test_files = []
    for case_num in test_cases:
        test_files.append(
            {
                "image": [
                    os.path.join(test_folder, case_num, f"{case_num}_flair.nii.gz"),
                    os.path.join(test_folder, case_num, f"{case_num}_t1ce.nii.gz"),
                    os.path.join(test_folder, case_num, f"{case_num}_t1.nii.gz"),
                    os.path.join(test_folder, case_num, f"{case_num}_t2.nii.gz"),
                ],
                "label": os.path.join(test_folder, case_num, f"{case_num}_seg.nii.gz"),
                "path": case_num,
            }
        )

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
