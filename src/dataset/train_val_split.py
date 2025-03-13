import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the existing cross-validation splits
split_path = "/home/magata/data/brats2021challenge/splits/data_splits.json"
with open(split_path, "r") as f:
    splits = json.load(f)

# Extract unique patients from training+validation folds
train_patients = set()
for sample in splits["training"]:
    train_patients.add(sample["image"][0].split("/")[-2])  # Extract patient ID

for sample in splits["validation"]:
    train_patients.add(sample["image"][0].split("/")[-2])  # Extract patient ID

print(f"Total Unique Patients in Training+Validation: {len(train_patients)}")

# Convert patient IDs to DataFrame
train_val_df = pd.DataFrame(list(train_patients), columns=["Patient"])

# Merge with full dataset info to retain region combination
data_info_path = "/home/magata/data/brats2021challenge/brats2021_class_presence_all.csv"
full_data = pd.read_csv(data_info_path)
train_val_df = train_val_df.merge(full_data, on="Patient")

# Count occurrences of each region combination
class_counts = train_val_df["Region Combination"].value_counts()
print(class_counts)

# Identify rare classes (with only one sample)
rare_classes = class_counts[class_counts == 1].index.tolist()
print(f"Rare Classes: {rare_classes}")

# Ensure all rare samples go to training set
rare_samples = train_val_df[train_val_df["Region Combination"].isin(rare_classes)]
train_val_df = train_val_df[~train_val_df["Region Combination"].isin(rare_classes)]  # Remove from main set

# Perform stratified split only on non-rare samples
train_patients, val_patients = train_test_split(
    train_val_df,
    test_size=0.1,
    random_state=42,
    stratify=train_val_df["Region Combination"]
)

# Add rare samples back to training set
train_patients = pd.concat([train_patients, rare_samples])

print(f"Final Training Set: {len(train_patients)} patients")
print(f"Final Validation Set: {len(val_patients)} patients")

final_splits = {"training": [], "validation": [], "test": splits["test"]}  # Keep test set unchanged

# Store training set
for _, row in train_patients.iterrows():
    final_splits["training"].append({
        "image": [
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_flair.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1ce.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t2.nii.gz"
        ],
        "label": f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_seg.nii.gz",
        "region_combination": row["Region Combination"]
    })

# Store validation set
for _, row in val_patients.iterrows():
    final_splits["validation"].append({
        "image": [
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_flair.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1ce.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t1.nii.gz",
            f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_t2.nii.gz"
        ],
        "label": f"RelabeledTrainingData/{row['Patient']}/{row['Patient']}_seg.nii.gz",
        "region_combination": row["Region Combination"]
    })

# Save the final training splits
final_split_path = "/home/magata/data/brats2021challenge/splits/final_training_splits.json"
with open(final_split_path, "w") as f:
    json.dump(final_splits, f, indent=4)

print(f"Final training splits saved to {final_split_path}")
