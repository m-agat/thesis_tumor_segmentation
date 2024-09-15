import os

def generate_filtered_folder_structure(start_path, prefix="", max_files_per_folder=5, max_subfolders_per_folder=5, depth_limit=3, current_depth=0, show_nii_in_folders=2):
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        return  # Skip directories where we don't have access

    pointers = ["> ", "└── "]
    
    # Filter out hidden files and directories
    items = [item for item in items if not item.startswith('.')]
    
    files = [item for item in items if os.path.isfile(os.path.join(start_path, item))]
    directories = [item for item in items if os.path.isdir(os.path.join(start_path, item))]

    # Always display these key files in the root directory
    important_files = ["train.py", "test.py", "submit_train.py", "README.md", "requirements.txt", "requirements_azure.txt"]
    
    # Show important files if we're in the root directory
    if current_depth == 0:
        for important_file in important_files:
            if important_file in files:
                print(f"{prefix}{pointers[0]}{important_file}")
        files = [file for file in files if file not in important_files]

    # Process directories first (show limited)
    for i, directory in enumerate(directories[:max_subfolders_per_folder]):
        pointer = pointers[1] if i == len(directories[:max_subfolders_per_folder]) - 1 and not files else pointers[0]
        print(f"{prefix}{pointer}{directory}")
        
        if current_depth < depth_limit:
            new_prefix = f"{prefix}    " if i == len(directories[:max_subfolders_per_folder]) - 1 else f"{prefix}│   "
            generate_filtered_folder_structure(os.path.join(start_path, directory), new_prefix, max_files_per_folder, max_subfolders_per_folder, depth_limit, current_depth + 1, show_nii_in_folders)

    if len(directories) > max_subfolders_per_folder:
        print(f"{prefix}└── ...")

    # Process files (show limited, but include .nii.gz in only some folders)
    nii_gz_files = [file for file in files if file.endswith('.nii.gz')]
    other_files = [file for file in files if not file.endswith('.nii.gz')]

    show_nii_files = show_nii_in_folders > 0
    if show_nii_files:
        # Show .nii.gz files in a few folders
        files_to_show = nii_gz_files[:max_files_per_folder]
        show_nii_in_folders -= 1
    else:
        files_to_show = []
    
    files_to_show += other_files[:max_files_per_folder]

    for i, file in enumerate(files_to_show):
        pointer = pointers[1] if i == len(files_to_show) - 1 else pointers[0]
        print(f"{prefix}{pointer}{file}")

    if len(other_files) > max_files_per_folder or (show_nii_files and len(nii_gz_files) > max_files_per_folder):
        print(f"{prefix}└── ...")

# Path to the directory where your dataset folder is stored
dataset_directory = "/home/agata/Desktop/thesis_tumor_segmentation"
generate_filtered_folder_structure(dataset_directory)
