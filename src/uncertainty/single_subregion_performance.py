import os
import nibabel as nib
import numpy as np
import pandas as pd

# Paths
outputs_dir = "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/SwinUNetr_Results_Test_Set"
csv_dir = "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr"
metrics_csv_path = os.path.join(csv_dir, "patient_scores_swinunetr.csv")

# Load performance metrics CSV
performance_data = pd.read_csv(metrics_csv_path)

# Initialize lists for new metrics
tc_metrics = {"Dice": [], "Composite_Score": [], "HD95": [], "Sensitivity": []}
wt_metrics = {"Dice": [], "Composite_Score": [], "HD95": [], "Sensitivity": []}
et_metrics = {"Dice": [], "Composite_Score": [], "HD95": [], "Sensitivity": []}


def safe_metric(value, default=0.0):
    """
    Safely handle invalid metric values.
    Replace NaN, inf, and -inf with a default value.
    """
    return np.nan_to_num(value, nan=default, posinf=default, neginf=default)


for _, row in performance_data.iterrows():
    patient_id = row["Patient"]
    seg_path = os.path.join(outputs_dir, f"{patient_id}_segmentation.nii.gz")

    # Load segmentation
    seg = nib.load(seg_path).get_fdata()

    # Count voxels for each class
    ncr_count = np.sum(seg == 1)
    ed_count = np.sum(seg == 2)
    et_count = np.sum(seg == 4)

    # Compute total voxel counts
    total_tc_count = ncr_count + et_count  # Tumor Core
    total_wt_count = ncr_count + ed_count + et_count  # Whole Tumor

    # Fetch and sanitize metrics for subregions
    dice_ncr, dice_ed, dice_et = map(
        lambda x: safe_metric(row[x]), ["Dice_1", "Dice_2", "Dice_4"]
    )
    composite_ncr, composite_ed, composite_et = map(
        lambda x: safe_metric(row[x]),
        ["Composite_Score_1", "Composite_Score_2", "Composite_Score_4"],
    )
    hd95_ncr, hd95_ed, hd95_et = map(
        lambda x: safe_metric(row[x], default=1e6), ["HD95_1", "HD95_2", "HD95_4"]
    )
    sensitivity_ncr, sensitivity_ed, sensitivity_et = map(
        lambda x: safe_metric(row[x]),
        ["Sensitivity_1", "Sensitivity_2", "Sensitivity_4"],
    )

    # Weighted metrics for Tumor Core
    if total_tc_count > 0:
        tc_metrics["Dice"].append(
            (dice_ncr * ncr_count + dice_et * et_count) / total_tc_count
        )
        tc_metrics["Composite_Score"].append(
            (composite_ncr * ncr_count + composite_et * et_count) / total_tc_count
        )
        tc_metrics["HD95"].append(
            (hd95_ncr * ncr_count + hd95_et * et_count) / total_tc_count
        )
        tc_metrics["Sensitivity"].append(
            (sensitivity_ncr * ncr_count + sensitivity_et * et_count) / total_tc_count
        )
    else:
        for key in tc_metrics:
            tc_metrics[key].append(0)

    # Weighted metrics for Whole Tumor
    if total_wt_count > 0:
        wt_metrics["Dice"].append(
            (dice_ncr * ncr_count + dice_ed * ed_count + dice_et * et_count)
            / total_wt_count
        )
        wt_metrics["Composite_Score"].append(
            (
                composite_ncr * ncr_count
                + composite_ed * ed_count
                + composite_et * et_count
            )
            / total_wt_count
        )
        wt_metrics["HD95"].append(
            (hd95_ncr * ncr_count + hd95_ed * ed_count + hd95_et * et_count)
            / total_wt_count
        )
        wt_metrics["Sensitivity"].append(
            (
                sensitivity_ncr * ncr_count
                + sensitivity_ed * ed_count
                + sensitivity_et * et_count
            )
            / total_wt_count
        )
    else:
        for key in wt_metrics:
            wt_metrics[key].append(0)

    # Metrics for Enhancing Tumor
    et_metrics["Dice"].append(dice_et)
    et_metrics["Composite_Score"].append(composite_et)
    et_metrics["HD95"].append(hd95_et)
    et_metrics["Sensitivity"].append(sensitivity_et)

# Create a new DataFrame with all computed metrics
updated_performance_data = pd.DataFrame(
    {
        "Patient": performance_data["Patient"],
        "Dice_TC": tc_metrics["Dice"],
        "Composite_Score_TC": tc_metrics["Composite_Score"],
        "HD95_TC": tc_metrics["HD95"],
        "Sensitivity_TC": tc_metrics["Sensitivity"],
        "Dice_WT": wt_metrics["Dice"],
        "Composite_Score_WT": wt_metrics["Composite_Score"],
        "HD95_WT": wt_metrics["HD95"],
        "Sensitivity_WT": wt_metrics["Sensitivity"],
        "Dice_ET": et_metrics["Dice"],
        "Composite_Score_ET": et_metrics["Composite_Score"],
        "HD95_ET": et_metrics["HD95"],
        "Sensitivity_ET": et_metrics["Sensitivity"],
    }
)

# Save to a new CSV
output_csv_path = os.path.join(csv_dir, "subregion_swinunetr_performance_full.csv")
updated_performance_data.to_csv(output_csv_path, index=False)

print(f"Subregion performance metrics (full) saved to {output_csv_path}")
