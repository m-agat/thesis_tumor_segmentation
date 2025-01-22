import os
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load the CSV file
data_info_path = "/home/magata/data/brats2021challenge/brats2021_class_presence_all.csv"
data = pd.read_csv(data_info_path)

# Separate data by region combination
region_groups = data.groupby('Region Combination')

# Reserve test samples
test_samples = []
remaining_data = []

# Ensure rare combinations are in the test set
for region, group in region_groups:
    if region in ['0-1-1', '1-1-0']:  # Rare combinations
        # Reserve at least 5 samples for rare combinations (or fewer if not enough)
        test_samples.append(group.sample(min(len(group), 5), random_state=42))
        remaining_data.append(group.drop(test_samples[-1].index))
    elif region == "0-1-0":
        test_samples.append(group.sample(1, random_state=42))
        remaining_data.append(group.drop(test_samples[-1].index))
    elif region == '1-1-1':  # Most common combination
        # Allocate 80% to train/val and 20% to test
        common_train, common_test = train_test_split(
            group,
            test_size=0.15,
            random_state=42,
            stratify=group['Region Combination']
        )
        test_samples.append(common_test)
        remaining_data.append(common_train)
    else:
        # Include all other rare combinations in the train set
        remaining_data.append(group)

# Combine test samples and remaining data
test = pd.concat(test_samples).reset_index(drop=True)
train_val = pd.concat(remaining_data).reset_index(drop=True)

# Perform stratified split on train/val data
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

folds_data = {"training": [], "validation": [], "test": []}

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val['Region Combination'])):
    train_patients = []
    val_patients = []
    
    for idx in train_idx:
        sample = train_val.iloc[idx]
        train_patients.append(sample["Patient"])
        folds_data["training"].append({
            "fold": fold,
            "image": [
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_flair.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t1ce.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t1.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t2.nii.gz"
            ],
            "label": f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_seg.nii.gz",
            "region_combination": sample["Region Combination"]
        })
    
    for idx in val_idx:
        sample = train_val.iloc[idx]
        val_patients.append(sample["Patient"])
        folds_data["validation"].append({
            "fold": fold,
            "image": [
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_flair.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t1ce.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t1.nii.gz",
                f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_t2.nii.gz"
            ],
            "label": f"RelabeledTrainingData/{sample['Patient']}/{sample['Patient']}_seg.nii.gz",
            "region_combination": sample["Region Combination"]
        })

    print(f"Fold {fold}:")
    print(f"  Train split: {len(train_patients)} patients")
    print(f"  Validation split: {len(val_patients)} patients")

# Add test set to folds_data
for idx, row in test.iterrows():
    folds_data["test"].append({
        "image": [
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_flair.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1ce.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t2.nii.gz"
        ],
        "label": f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_seg.nii.gz",
        "region_combination": row["Region Combination"]
    })

# Save to JSON file
output_dir = "/home/magata/data/brats2021challenge/splits/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "data_splits.json")
with open(output_path, "w") as f:
    json.dump(folds_data, f, indent=4)

print(f"Data splits saved to {output_path}")
print(f"Test set size: {len(folds_data['test'])}")
print(f"Training and validation folds: {n_splits}")
