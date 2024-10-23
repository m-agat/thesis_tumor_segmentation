import os
import shutil
import random

"""
Split the BraTS2021 dataset into train-val-test 

- training cases = 875 (70%)
- validation cases = 188 (15%)
- test cases = 188 (15%)

"""

random.seed(42)

data_dir = (
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/TrainingData"
)
train_dir = (
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/train"
)
val_dir = (
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/val"
)
test_dir = (
    "/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/test"
)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

cases = [case for case in os.listdir(data_dir) if case.startswith("BraTS2021")]

# Shuffle cases for random splitting
random.shuffle(cases)

# Split data
train_split = int(0.7 * len(cases))
val_split = int(0.85 * len(cases))

train_cases = cases[:train_split]
val_cases = cases[train_split:val_split]
test_cases = cases[val_split:]


def copy_cases(cases, dest_dir):
    for case in cases:
        case_path = os.path.join(data_dir, case)
        dest_case_path = os.path.join(dest_dir, case)
        shutil.copytree(case_path, dest_case_path)


copy_cases(train_cases, train_dir)
copy_cases(val_cases, val_dir)
copy_cases(test_cases, test_dir)

print(f"Total cases: {len(cases)}")
print(f"Training cases: {len(train_cases)}")
print(f"Validation cases: {len(val_cases)}")
print(f"Testing cases: {len(test_cases)}")
