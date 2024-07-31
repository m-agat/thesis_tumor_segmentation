import os
import shutil

def create_subset(original_dir, new_dir, num_cases):
    # Create the new directory if it doesn't exist
    os.makedirs(new_dir, exist_ok=True)

    # List all cases in the original directory
    cases = os.listdir(original_dir)

    # Copy a subset of cases to the new directory
    for i, case in enumerate(cases):
        if i >= num_cases:
            break
        case_path = os.path.join(original_dir, case)
        new_case_path = os.path.join(new_dir, case)
        shutil.copytree(case_path, new_case_path)

    print(f"Copied {num_cases} cases to {new_dir}")

# Define the original and new directories and the number of cases to copy
# original_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2"
# new_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-TrainingData/training_data_subset"
original_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-ValidationData/validation_data"
new_dir = f"{os.getcwd()}/BraTS2024-BraTS-GLI-ValidationData/validation_data_subset"
num_cases = 3  # Number of cases to copy

# Create the subset
create_subset(original_dir, new_dir, num_cases)
