import os
import nibabel as nib
import numpy as np
import pandas as pd

# Paths
outputs_dir = (
    "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/uncertainty/outputs"
)
metrics_csv_path = os.path.join(outputs_dir, "ensemble_performance_no_vnet.csv")

# Load performance metrics CSV
performance_data = pd.read_csv(metrics_csv_path)

# Initialize lists for new metrics
tc_metrics = {
    "Dice": [],
    "Composite_Score": [],
    "HD95": [],
    "F1": [],
    "Sensitivity": [],
}
wt_metrics = {
    "Dice": [],
    "Composite_Score": [],
    "HD95": [],
    "F1": [],
    "Sensitivity": [],
}
et_metrics = {
    "Dice": [],
    "Composite_Score": [],
    "HD95": [],
    "F1": [],
    "Sensitivity": [],
}


def safe_metric(metric_value, default=0.0):
    """
    Safely handle invalid metric values.
    Replace NaN, inf, and -inf with a default value.
    """
    return np.nan_to_num(metric_value, nan=default, posinf=default, neginf=default)


for _, row in performance_data.iterrows():
    patient_id = row["Patient"]
    seg_path = os.path.join(outputs_dir, f"{patient_id}_ensemble_segmentation.nii.gz")

    # Load segmentation
    seg = nib.load(seg_path).get_fdata()

    # Count voxels for each class
    ncr_count = np.sum(seg == 1)
    ed_count = np.sum(seg == 2)
    et_count = np.sum(seg == 4)

    # Compute total voxel counts
    total_tc_count = ncr_count + et_count  # Tumor Core
    total_wt_count = ncr_count + ed_count + et_count  # Whole Tumor

    # Fetch metrics for subregions, replacing NaN and infinities
    dice_ncr, dice_ed, dice_et = map(
        lambda x: safe_metric(row[x]), ["Dice_1", "Dice_2", "Dice_4"]
    )
    composite_ncr, composite_ed, composite_et = map(
        lambda x: safe_metric(row[x]),
        ["Composite_Score_1", "Composite_Score_2", "Composite_Score_4"],
    )
    hd95_ncr, hd95_ed, hd95_et = map(
        lambda x: safe_metric(row[x], default=1e6), ["HD95_1", "HD95_2", "HD95_4"]
    )  # Use a large value for HD95
    f1_ncr, f1_ed, f1_et = map(lambda x: safe_metric(row[x]), ["F1_1", "F1_2", "F1_4"])
    sensitivity_ncr, sensitivity_ed, sensitivity_et = map(
        lambda x: safe_metric(row[x]),
        ["Sensitivity_1", "Sensitivity_2", "Sensitivity_4"],
    )

    # Normalize voxel counts to prevent overflow
    total_voxels = ncr_count + ed_count + et_count
    if total_voxels > 0:
        ncr_ratio = ncr_count / total_voxels
        ed_ratio = ed_count / total_voxels
        et_ratio = et_count / total_voxels
    else:
        ncr_ratio, ed_ratio, et_ratio = 0, 0, 0

    # Weighted metrics for Tumor Core (NCR + ET)
    if total_tc_count > 0:
        for metric, ncr, et in [
            ("Dice", dice_ncr, dice_et),
            ("Composite_Score", composite_ncr, composite_et),
            ("HD95", hd95_ncr, hd95_et),
            ("F1", f1_ncr, f1_et),
            ("Sensitivity", sensitivity_ncr, sensitivity_et),
        ]:
            weighted_tc_metric = (ncr * ncr_ratio + et * et_ratio) / (
                ncr_ratio + et_ratio
            )
            tc_metrics[metric].append(weighted_tc_metric)
    else:
        for metric in tc_metrics:
            tc_metrics[metric].append(0)

    # Weighted metrics for Whole Tumor (NCR + ED + ET)
    if total_wt_count > 0:
        for metric, ncr, ed, et in [
            ("Dice", dice_ncr, dice_ed, dice_et),
            ("Composite_Score", composite_ncr, composite_ed, composite_et),
            ("HD95", hd95_ncr, hd95_ed, hd95_et),
            ("F1", f1_ncr, f1_ed, f1_et),
            ("Sensitivity", sensitivity_ncr, sensitivity_ed, sensitivity_et),
        ]:
            weighted_wt_metric = (ncr * ncr_ratio + ed * ed_ratio + et * et_ratio) / (
                ncr_ratio + ed_ratio + et_ratio
            )
            wt_metrics[metric].append(weighted_wt_metric)
    else:
        for metric in wt_metrics:
            wt_metrics[metric].append(0)

    # Metrics for Enhancing Tumor (ET)
    for metric, et in [
        ("Dice", dice_et),
        ("Composite_Score", composite_et),
        ("HD95", hd95_et),
        ("F1", f1_et),
        ("Sensitivity", sensitivity_et),
    ]:
        et_metrics[metric].append(et)

# Create a new DataFrame with the computed metrics
updated_performance_data = pd.DataFrame(
    {
        "Patient": performance_data["Patient"],
        "Dice_TC": tc_metrics["Dice"],
        "Composite_Score_TC": tc_metrics["Composite_Score"],
        "HD95_TC": tc_metrics["HD95"],
        "F1_TC": tc_metrics["F1"],
        "Sensitivity_TC": tc_metrics["Sensitivity"],
        "Dice_WT": wt_metrics["Dice"],
        "Composite_Score_WT": wt_metrics["Composite_Score"],
        "HD95_WT": wt_metrics["HD95"],
        "F1_WT": wt_metrics["F1"],
        "Sensitivity_WT": wt_metrics["Sensitivity"],
        "Dice_ET": et_metrics["Dice"],
        "Composite_Score_ET": et_metrics["Composite_Score"],
        "HD95_ET": et_metrics["HD95"],
        "F1_ET": et_metrics["F1"],
        "Sensitivity_ET": et_metrics["Sensitivity"],
    }
)

# Save to a new CSV
output_csv_path = os.path.join(outputs_dir, "subregion_ensemble_performance_full.csv")
updated_performance_data.to_csv(output_csv_path, index=False)

print(f"Subregion performance metrics (full) saved to {output_csv_path}")
