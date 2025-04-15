import pandas as pd
import numpy as np

# Load the ground truth file with region presence information.
# The CSV should have columns: Patient, NCR present, ED present, ET present, Region Combination.
# For this example, we assume the CSV is named 'ground_truth_regions.csv'
gt = pd.read_csv(r'\Users\agata\Desktop\thesis_tumor_segmentation\EDA\brats2021_class_presence_all.csv')

# Format the patient IDs as needed (assume they are in a comparable format).
# If the Patient column is already in a 5-digit format, you can skip this step.
gt['patient_id'] = gt['Patient'].apply(lambda x: x.split('_')[-1])

# Load the prediction metrics for each model.
vnet = pd.read_csv('../models/performance/vnet/patient_metrics_test_vnet.csv')
att = pd.read_csv('../models/performance/attunet/patient_metrics_test_attunet.csv')
seg = pd.read_csv('../models/performance/segresnet/patient_metrics_test_segresnet.csv')
swin = pd.read_csv('../models/performance/swinunetr/patient_metrics_test_swinunetr.csv')

# Format the patient_id column as a 5-digit string.
for df in [vnet, att, seg, swin]:
    df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")

# Merge each model's results with the ground truth on patient_id.
vnet_merged = pd.merge(vnet, gt, on='patient_id')
att_merged = pd.merge(att, gt, on='patient_id')
seg_merged = pd.merge(seg, gt, on='patient_id')
swin_merged = pd.merge(swin, gt, on='patient_id')

# For each sub-region, filter for patients where the region is absent (value 0 in ground truth)
# Then compute the mean Dice score for that sub-region.
def compute_absence_performance(model_df, region_label):
    # region_label should be like 'NCR present', 'ED present', or 'ET present'
    # Filter patients where that region is absent (i.e., value==0)
    absent_df = model_df[model_df[region_label] == 0]
    # Assume the Dice score column is labeled "Dice NCR", "Dice ED", or "Dice ET" accordingly.
    dice_col = "Dice " + region_label.split()[0]  # e.g., "Dice NCR" if region_label is "NCR present"
    mean_dice = np.nanmean(absent_df[dice_col])
    return mean_dice, len(absent_df)

# Calculate for each model and for each region:
regions = ['NCR present', 'ED present', 'ET present']
models = {'V-Net': vnet_merged, 'Attention UNet': att_merged, 'SegResNet': seg_merged, 'SwinUNETR': swin_merged}

print("Model robustness in predicting absence (Dice score on absent cases):")
for region in regions:
    print(f"\nFor {region.split()[0]}:")
    for model_name, model_df in models.items():
        mean_dice, count = compute_absence_performance(model_df, region)
        print(f"  {model_name} (n={count}): mean Dice = {mean_dice:.3f}")
