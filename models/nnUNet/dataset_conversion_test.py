import os
import shutil

# Paths
validation_data_dir = '/home/agata/Desktop/thesis_tumor_segmentation/data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
output_imagesTs_dir = '/home/agata/Desktop/thesis_tumor_segmentation/models/nnUNet/nnUNet_raw/Dataset100_BraTS2023/imagesTs'

# Create imagesTs folder if it doesn't exist
os.makedirs(output_imagesTs_dir, exist_ok=True)

# Loop through the validation cases and copy the modalities to imagesTs
case_ids = [f for f in os.listdir(validation_data_dir) if os.path.isdir(os.path.join(validation_data_dir, f))]

for case_id in case_ids:
    case_path = os.path.join(validation_data_dir, case_id)
    
    # Copy each modality (T1c, T1n, T2w, T2f) and rename them with _0000, _0001, _0002, _0003 suffixes
    shutil.copy(os.path.join(case_path, case_id + "-t1n.nii.gz"), os.path.join(output_imagesTs_dir, case_id + "_0000.nii.gz"))
    shutil.copy(os.path.join(case_path, case_id + "-t1c.nii.gz"), os.path.join(output_imagesTs_dir, case_id + "_0001.nii.gz"))
    shutil.copy(os.path.join(case_path, case_id + "-t2w.nii.gz"), os.path.join(output_imagesTs_dir, case_id + "_0002.nii.gz"))
    shutil.copy(os.path.join(case_path, case_id + "-t2f.nii.gz"), os.path.join(output_imagesTs_dir, case_id + "_0003.nii.gz"))

print(f"Copied validation data to {output_imagesTs_dir}")
