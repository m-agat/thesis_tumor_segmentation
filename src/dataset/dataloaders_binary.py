import os
from monai import transforms, data
from dataset.transforms import train_transform_wt_binary, val_transform_wt_binary

def read_data_from_folders_binary_wt(train_folder, val_folder):
    """
    Read training and validation data for binary WT segmentation from specific folders.
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
    train_files = load_cases_from_folder(train_folder)
    val_files = load_cases_from_folder(val_folder)    
    return train_files, val_files


def get_loader_wt_binary(batch_size, train_folder, val_folder, global_roi):
    """
    Create data loaders for WT binary segmentation.
    """
    # Load train and validation files
    train_files, val_files = read_data_from_folders_binary_wt(train_folder, val_folder)

    # Binary WT transforms (no need for local/global at this stage)
    global_transform_wt_binary = train_transform_wt_binary(global_roi=global_roi)

    val_transform_wt_bin = val_transform_wt_binary()

    # Create datasets with WT binary labels
    train_ds = data.Dataset(data=train_files, transform=global_transform_wt_binary)
    val_ds = data.Dataset(data=val_files, transform=val_transform_wt_bin)

    # Create data loaders
    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


