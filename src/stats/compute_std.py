import pandas as pd
import numpy as np

# Load patient-level performance data for each model
vnet_df = pd.read_csv('../models/performance/vnet/patient_metrics_test_vnet.csv')
att_df = pd.read_csv('../models/performance/attunet/patient_metrics_test_attunet.csv')
seg_df = pd.read_csv('../models/performance/segresnet/patient_metrics_test_segresnet.csv')
swin_df = pd.read_csv('../models/performance/swinunetr/patient_metrics_test_swinunetr.csv')

# Format the patient_id column as a 5-digit string.
for df in [att_df, seg_df, swin_df, vnet_df]:
    df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")

# Create a dictionary to store model names and their dataframes
model_dfs = {
    'V-Net': vnet_df,
    'Attention U-Net': att_df,
    'SegResNet': seg_df,
    'SwinUNETR': swin_df
}
# df = pd.read_csv('../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv')
# Define metrics to average (excluding HD95)
metrics_to_average = ['Dice', 'Sensitivity', 'Specificity', 'HD95']
sub_regions = ['NCR', 'ED', 'ET']

# Add averaged columns for each dataframe
for model_name, df in model_dfs.items():
    for metric in metrics_to_average:
        # Create the new column name for average across regions
        new_col_avg = f'{metric} avg_regions'
        # Compute average of NCR, ED, and ET for each metric
        df[new_col_avg] = df[[f'{metric} {region}' for region in sub_regions]].mean(axis=1)

# Compute and print average and standard deviation for each model and sub-region
for model_name, df in model_dfs.items():
    print(f"\nPerformance of {model_name}:")
    for metric in metrics_to_average:
        for region in sub_regions:
            mean_val = df[f'{metric} {region}'].mean()
            std_val = df[f'{metric} {region}'].std()
            print(f"  {metric} {region}: Avg = {mean_val:.4f}, STD = {std_val:.4f}")
        # Print average across regions as well
        mean_avg = df[f'{metric} avg_regions'].mean()
        std_avg = df[f'{metric} avg_regions'].std()
        print(f"  {metric} avg_regions: Avg = {mean_avg:.4f}, STD = {std_avg:.4f}")