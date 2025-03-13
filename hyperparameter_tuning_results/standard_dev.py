from tbparse import SummaryReader
import numpy as np
import os

# AdamW_lr_0.0001_wd_1e-05
# SGD_lr_0.001_wd_0.0001
# Define main folder path and subdirectories for each fold
main_folder_path = "./segresnet/SGD_lr_0.001_wd_0.0001"
fold_dirs = ["fold1", "fold2", "fold3", "fold4", "fold5"]
metrics = ["Dice", "HD95", "Sensitivity", "Specificity"]

for metric in metrics:
    # Initialize lists to store last recorded validation scores per fold
    ncr_scores = []
    ed_scores = []
    et_scores = []

    # Loop through each fold
    for i, fold_dir in enumerate(fold_dirs, start=1):
        fold_dir_full_path = os.path.join(main_folder_path, fold_dir)
        reader = SummaryReader(fold_dir_full_path, pivot=True)

        # Extract only the last available validation scores for each tumor sub-region
        ncr_key = f"Fold_{i}/{metric}/Validation_NCR"
        ed_key = f"Fold_{i}/{metric}/Validation_ED"
        et_key = f"Fold_{i}/{metric}/Validation_ET"

        if ncr_key in reader.scalars.columns:
            ncr_values = reader.scalars[ncr_key].dropna().tolist()
            if ncr_values:
                ncr_scores.append(ncr_values[-1])  # Store only the last value

        if ed_key in reader.scalars.columns:
            ed_values = reader.scalars[ed_key].dropna().tolist()
            if ed_values:
                ed_scores.append(ed_values[-1])

        if et_key in reader.scalars.columns:
            et_values = reader.scalars[et_key].dropna().tolist()
            if et_values:
                et_scores.append(et_values[-1])

    # Convert lists to NumPy arrays
    ncr_scores = np.array(ncr_scores)
    ed_scores = np.array(ed_scores)
    et_scores = np.array(et_scores)

    # Compute standard deviation across folds
    std_ncr = np.nanstd(ncr_scores)
    std_ed = np.nanstd(ed_scores)
    std_et = np.nanstd(et_scores)

    # Print results
    print(f"Cross-validation {metric} NCR STD: {std_ncr:.4f}")
    print(f"Cross-validation {metric} ED STD: {std_ed:.4f}")
    print(f"Cross-validation {metric} ET STD: {std_et:.4f}")

    # Combine all HD95 scores across folds and sub-regions
    all_scores = np.concatenate([ncr_scores, ed_scores, et_scores])

    # Compute standard deviation across all scores
    overall_std = np.nanstd(all_scores)

    print(f"Overall Cross-validation {metric} STD: {overall_std:.4f}\n")
