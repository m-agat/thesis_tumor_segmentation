import os
import shutil

# Source folder where nnUNet predictions are stored
nnunet_predictions_dir = "/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_predictions/segmentations100"

# Target folder where all models' predictions are organized
target_dir = "/home/agata/Desktop/thesis_tumor_segmentation/results/segmentations"

# Loop over all files in the nnUNet predictions folder
for filename in os.listdir(nnunet_predictions_dir):
    if filename.endswith(".nii.gz"):
        # Extract the case number from the file (e.g., "BraTS2021_01619" from "BraTS2021_01619.nii.gz")
        case_num = filename.split(".")[0]  # "BraTS2021_01619"
        
        # Define the target folder for this case
        case_folder = os.path.join(target_dir, case_num)
        
        # Create the folder if it doesn't exist
        if not os.path.exists(case_folder):
            os.makedirs(case_folder)

        # Define the source file path
        src_path = os.path.join(nnunet_predictions_dir, filename)

        # Define the target file path in the case folder, following the naming convention used for other models
        target_path = os.path.join(case_folder, f"{case_num}_nnunet_segmentation.nii.gz")

        # Move the file to the target folder
        shutil.move(src_path, target_path)
        print(f"Moved {src_path} to {target_path}")

print("All nnUNet predictions have been moved.")
