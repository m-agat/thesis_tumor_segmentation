import pandas as pd
import numpy as np

# Load patient-level performance data for each model
att_df = pd.read_csv('../models/performance/attunet/patient_metrics_test_attunet.csv')
seg_df = pd.read_csv('../models/performance/segresnet/patient_metrics_test_segresnet.csv')
swin_df = pd.read_csv('../models/performance/swinunetr/patient_metrics_test_swinunetr.csv')
vnet_df = pd.read_csv('../models/performance/vnet/patient_metrics_test_vnet.csv')

# Format the patient_id column as a 5-digit string.
for df in [att_df, seg_df, swin_df, vnet_df]:
    df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")

# Create a combined DataFrame with the metrics of interest (including patient_id)
data = pd.DataFrame({
    'patient_id': att_df['patient_id'],
    'Att_DiceOverall': att_df['Dice overall'],
    'Seg_DiceOverall': seg_df['Dice overall'],
    'Swin_DiceOverall': swin_df['Dice overall'],
    'VNet_DiceOverall': vnet_df['Dice overall'],
    'Att_DiceNCR': att_df['Dice NCR'],
    'Seg_DiceNCR': seg_df['Dice NCR'],
    'Swin_DiceNCR': swin_df['Dice NCR'],
    'VNet_DiceNCR': vnet_df['Dice NCR']
})

# --- 1. Find cases where one model is significantly better in Dice NCR than the others ---

# Filter out cases where any model has Dice NCR equal to 1 (likely indicating absence of the NCR region)
filtered_data = data[(data['Att_DiceNCR'] < 1.0) & (data['Seg_DiceNCR'] < 1.0) &
                     (data['Swin_DiceNCR'] < 1.0) & (data['VNet_DiceNCR'] < 1.0)]

def get_ordered_metrics(row, keys):
    # Create a dictionary of model names and their corresponding Dice NCR values.
    values_dict = {k: row[k] for k in keys}
    # Sort model names based on their values.
    sorted_models = sorted(values_dict, key=lambda k: values_dict[k])
    best_model = sorted_models[-1]
    second_best_model = sorted_models[-2]
    best = values_dict[best_model]
    second_best = values_dict[second_best_model]
    diff = best - second_best
    return best, second_best, diff, best_model, second_best_model

keys_ncr = ['Att_DiceNCR', 'Seg_DiceNCR', 'Swin_DiceNCR', 'VNet_DiceNCR']

# Apply row-wise: compute the best and second-best Dice NCR values, their difference, and record the corresponding models.
filtered_data[['Best_DiceNCR', 'SecondBest_DiceNCR', 'Diff_DiceNCR',
               'Best_Model', 'Second_Best_Model']] = filtered_data.apply(
    lambda row: pd.Series(get_ordered_metrics(row, keys_ncr)), axis=1)

# Define a threshold for a significant difference (e.g., at least 0.10)
threshold_diff = 0.10
signif_cases = filtered_data[filtered_data['Diff_DiceNCR'] >= threshold_diff]

print("Cases where one model significantly outperforms the others in Dice NCR:")
print(signif_cases[['patient_id', 'Best_DiceNCR', 'Best_Model', 'SecondBest_DiceNCR', 'Second_Best_Model', 'Diff_DiceNCR']])

# --- 2. Find cases where all models are inconsistent (high variance in Dice overall) ---

# Compute the variance of Dice overall across the four models for each patient.
data['DiceOverall_Var'] = data[['Att_DiceOverall', 'Seg_DiceOverall', 'Swin_DiceOverall', 'VNet_DiceOverall']].var(axis=1)

# Define inconsistency as those in the top 10th percentile of variance.
variance_threshold = data['DiceOverall_Var'].quantile(0.90)
inconsistent_cases = data[data['DiceOverall_Var'] >= variance_threshold]

print("\nCases with high inconsistency in Dice overall (top 10th percentile of variance):")
print(inconsistent_cases[['patient_id', 'Att_DiceOverall', 'Seg_DiceOverall', 'Swin_DiceOverall', 'VNet_DiceOverall', 'DiceOverall_Var']])