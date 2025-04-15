import pandas as pd
import numpy as np

# Load patient-level performance data for each model
att_df = pd.read_csv('../models/performance/attunet/patient_metrics_test_attunet.csv')
seg_df = pd.read_csv('../models/performance/segresnet/patient_metrics_test_segresnet.csv')
swin_df = pd.read_csv('../models/performance/swinunetr/patient_metrics_test_swinunetr.csv')

# Format the patient_id column as a 5-digit string.
for df in [att_df, seg_df, swin_df]:
    df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")

# Create a combined DataFrame with the metrics of interest (including patient_id)
data = pd.DataFrame({
    'patient_id': att_df['patient_id'],
    'Att_DiceOverall': att_df['Dice overall'],
    'Seg_DiceOverall': seg_df['Dice overall'],
    'Swin_DiceOverall': swin_df['Dice overall'],
    'Att_DiceED': att_df['Dice ED'],
    'Seg_DiceED': seg_df['Dice ED'],
    'Swin_DiceED': swin_df['Dice ED']
})

# ------------------------------------------------------------------------
# CASE 1 & CASE 2: Find cases where one model's Dice ED is significantly higher
# than the other two.
# ------------------------------------------------------------------------
#
# Filter out cases where any model has Dice ED equal to 1 (likely indicating absence
# of the ED region)
filtered_data = data[(data['Att_DiceED'] < 1.0) &
                     (data['Seg_DiceED'] < 1.0) &
                     (data['Swin_DiceED'] < 1.0)]

def ordered_metrics_ED(row, keys):
    """
    For a row, return:
    - best value, second best value, their difference,
    - best model key, and second best model key.
    """
    values_dict = {k: row[k] for k in keys}
    # Sort keys by the corresponding metric (lowest to highest)
    sorted_models = sorted(values_dict, key=lambda k: values_dict[k])
    best_model = sorted_models[-1]
    second_best_model = sorted_models[-2]
    best = values_dict[best_model]
    second_best = values_dict[second_best_model]
    diff = best - second_best
    return best, second_best, diff, best_model, second_best_model

keys_ED = ['Att_DiceED', 'Seg_DiceED', 'Swin_DiceED']

# Apply row-wise: compute ordered metrics for Dice ED.
filtered_data[['Best_DiceED', 'SecondBest_DiceED', 'Diff_DiceED',
               'Best_Model', 'Second_Best_Model']] = filtered_data.apply(
    lambda row: pd.Series(ordered_metrics_ED(row, keys_ED)), axis=1)

# Define a threshold for a significant difference (e.g., at least 0.10)
threshold_diff = 0.10
signif_cases = filtered_data[filtered_data['Diff_DiceED'] >= threshold_diff]

# --- Case 1: Ideal case for Attention UNet in ED ---
# Filter the significant cases for which Attention UNet is best.
ideal_att_cases = signif_cases[signif_cases['Best_Model'] == 'Att_DiceED']

if not ideal_att_cases.empty:
    # Option: choose the case with the largest difference
    ideal_att = ideal_att_cases.sort_values('Diff_DiceED', ascending=False).iloc[0]
    print("Ideal Case for Attention UNet (ED):")
    print(ideal_att[['patient_id', 'Att_DiceED', 'SecondBest_DiceED', 'Diff_DiceED']])
else:
    print("No ideal Attention UNet case found based on Dice ED with threshold difference.")

# --- Case 2: Ideal case for SegResNet (using ED as proxy) ---
# Filter the significant cases for which SegResNet is best.
ideal_seg_cases = signif_cases[signif_cases['Best_Model'] == 'Seg_DiceED']

if not ideal_seg_cases.empty:
    ideal_seg = ideal_seg_cases.sort_values('Diff_DiceED', ascending=False).iloc[0]
    print("\nIdeal Case for SegResNet (ED proxy):")
    print(ideal_seg[['patient_id', 'Seg_DiceED', 'SecondBest_DiceED', 'Diff_DiceED']])
else:
    print("No ideal SegResNet case found based on Dice ED with threshold difference.")

# ------------------------------------------------------------------------
# CASE 3: Find a borderline/challenging case with high overall inconsistency.
# ------------------------------------------------------------------------
#
# Compute the variance (across models) of overall Dice for each patient.
data['DiceOverall_Var'] = data[['Att_DiceOverall', 'Seg_DiceOverall', 'Swin_DiceOverall']].var(axis=1)

# Define the top 10th percentile of variance as indicating high inconsistency.
variance_threshold = data['DiceOverall_Var'].quantile(0.90)
inconsistent_cases = data[data['DiceOverall_Var'] >= variance_threshold]

if not inconsistent_cases.empty:
    # You might select the case with the highest variance as the most challenging.
    borderline_case = inconsistent_cases.sort_values('DiceOverall_Var', ascending=False).iloc[0]
    print("\nBorderline (challenging) Case with high inconsistency in overall Dice:")
    print(borderline_case[['patient_id', 'Att_DiceOverall', 'Seg_DiceOverall', 'Swin_DiceOverall', 'DiceOverall_Var']])
else:
    print("No inconsistent cases found based on overall Dice variance (top 10th percentile).")
