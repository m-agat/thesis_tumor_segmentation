import os
from monai import transforms, data
from transforms import get_global_transforms, get_local_transforms, get_val_transforms, get_test_transforms


def read_data_from_folders(train_folder, val_folder, basedir):
    """
    Read training and validation data from specific folders.
    """
    def load_cases_from_folder(folder):
        cases = [f for f in os.listdir(folder) if f.startswith("BraTS2021")]
        files = []
        for case in cases:
            files.append({
                "image": [
                    os.path.join(folder, case, f"{case}_flair.nii.gz"),
                    os.path.join(folder, case, f"{case}_t1ce.nii.gz"),
                    os.path.join(folder, case, f"{case}_t1.nii.gz"),
                    os.path.join(folder, case, f"{case}_t2.nii.gz")
                ],
                "label": os.path.join(folder, case, f"{case}_seg.nii.gz")
            })
        return files

    # Load train and validation files
    train_files = load_cases_from_folder(os.path.join(basedir, train_folder))
    val_files = load_cases_from_folder(os.path.join(basedir, val_folder))
    
    return train_files, val_files


def get_loader(batch_size, data_dir, train_folder, val_folder, global_roi, local_roi):
    """
    Create data loaders for training with both global and local patches.
    """
    # Load train and validation files
    train_files, val_files = read_data_from_folders(train_folder, val_folder, data_dir)

    # Get the transforms from transforms.py
    global_transform = get_global_transforms(global_roi)
    local_transform = get_local_transforms(local_roi)
    val_transform = get_val_transforms()

    # Create datasets with global and local patches
    global_train_ds = data.Dataset(data=train_files, transform=global_transform)
    local_train_ds = data.Dataset(data=train_files, transform=local_transform)
    val_ds = data.Dataset(data=val_files, transform=val_transform)

    # Create data loaders
    global_train_loader = data.DataLoader(
        global_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    local_train_loader = data.DataLoader(
        local_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return global_train_loader, local_train_loader, val_loader


def load_test_data(test_folder, data_dir, batch_size=1):
    """
    Load all test files from a specified test folder and apply transformations.
    """
    # List all cases in the test folder
    test_cases = [f for f in os.listdir(os.path.join(data_dir, test_folder)) if f.startswith("BraTS2021")]
    
    test_files = []
    for case_num in test_cases:
        test_files.append({
            "image": [
                os.path.join(data_dir, test_folder, case_num, f"{case_num}_flair.nii.gz"),
                os.path.join(data_dir, test_folder, case_num, f"{case_num}_t1ce.nii.gz"),
                os.path.join(data_dir, test_folder, case_num, f"{case_num}_t1.nii.gz"),
                os.path.join(data_dir, test_folder, case_num, f"{case_num}_t2.nii.gz"),
            ],
            "label": os.path.join(data_dir, test_folder, case_num, f"{case_num}_seg.nii.gz")
        })
    
    # Get the test transforms
    test_transform = get_test_transforms()

    # Create test dataset
    test_ds = data.Dataset(data=test_files, transform=test_transform)

    # Create test DataLoader
    test_loader = data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    return test_loader
