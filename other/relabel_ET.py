import os
import nibabel as nib
import numpy as np
import shutil

# Input and output directories
input_dir = r"/home/magata/data/brats2021challenge/TrainingData"
output_dir = r"/home/magata/data/brats2021challenge/RelabeledTrainingData"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for idx, case_folder in enumerate(os.listdir(input_dir)):
    print(f"Case processed: {case_folder}, {idx}/{len(os.listdir(input_dir))}.")
    case_path = os.path.join(input_dir, case_folder)

    # Skip if it's not a folder
    if not os.path.isdir(case_path):
        continue

    # Create the corresponding folder in the output directory
    output_case_folder = os.path.join(output_dir, case_folder)
    os.makedirs(output_case_folder, exist_ok=True)

    # Iterate over all files in the case folder
    for filename in os.listdir(case_path):
        file_path = os.path.join(case_path, filename)

        if filename.endswith("_seg.nii.gz"):
            # Process segmentation file
            seg_image = nib.load(file_path)
            seg_data = seg_image.get_fdata()

            # Replace label 4 (Enhancing Tumor) with label 3
            seg_data[seg_data == 4] = 3

            # Normalize floating-point imprecision
            seg_data = np.round(seg_data).astype(int)

            # Validate unique values
            unique_values = np.unique(seg_data)
            if not np.array_equal(unique_values, [0, 1, 2, 3]):
                print(f"Warning: Unexpected labels {unique_values} found in {filename}")

            # Save the modified segmentation file
            output_seg_file = os.path.join(output_case_folder, filename)
            modified_seg = nib.Nifti1Image(seg_data, seg_image.affine, seg_image.header)
            nib.save(modified_seg, output_seg_file)

            print(f"Processed segmentation: {file_path} -> {output_seg_file}")
        else:
            # Copy non-segmentation files (e.g., MRI scans) to the output folder
            output_file = os.path.join(output_case_folder, filename)
            shutil.copy(file_path, output_file)
            print(f"Copied: {file_path} -> {output_file}")
