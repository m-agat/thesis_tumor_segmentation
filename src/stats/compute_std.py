import pandas as pd
import numpy as np

# # Load patient-level performance data for each model
# vnet_df = pd.read_csv('../models/performance/vnet/patient_metrics_test_vnet.csv')
# att_df = pd.read_csv('../models/performance/attunet/patient_metrics_test_attunet.csv')
# seg_df = pd.read_csv('../models/performance/segresnet/patient_metrics_test_segresnet.csv')
# swin_df = pd.read_csv('../models/performance/swinunetr/patient_metrics_test_swinunetr.csv')

# # Format the patient_id column as a 5-digit string.
# for df in [att_df, seg_df, swin_df, vnet_df]:
#     df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")

# # Create a dictionary to store model names and their dataframes
# model_dfs = {
#     'V-Net': vnet_df,
#     'Attention U-Net': att_df,
#     'SegResNet': seg_df,
#     'SwinUNETR': swin_df
# }
df = pd.read_csv('../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv')
# Define metrics to average (excluding HD95)
metrics_to_average = ['Dice', 'Sensitivity', 'Specificity', 'HD95']

# Add averaged columns for each dataframe
# for model_name, df in model_dfs.items():
for metric in metrics_to_average:
    # Create the new column name
    new_col = f'{metric} avg_regions'
    # Compute average of NCR, ED, and ET for each metric
    df[new_col] = df[[f'{metric} NCR', f'{metric} ED', f'{metric} ET']].mean(axis=1)

# Compute standard deviation for each column in each model's dataframe
# for model_name, df in model_dfs.items():
# print(f"\nStandard deviations for {model_name}:")
# Get numerical columns only (excluding patient_id which is now string)
numerical_cols = df.select_dtypes(include=[np.number]).columns
std_values = df[numerical_cols].std()

for metric in metrics_to_average:
    mean = df[f'{metric} avg_regions'].mean()
    print(f"{metric}: {mean:.4f}")

#rint standard deviations with formatted output
for col in numerical_cols:
    print(f"{col}: {std_values[col]:.4f}")