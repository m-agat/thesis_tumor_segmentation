import pandas as pd
import numpy as np

metric = "ET"
# Load patient-level performance data for each model
simple_avg_df = pd.read_csv('../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv')
perf_weight_df = pd.read_csv('../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv')
ttd_df = pd.read_csv('../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv')
tta_df = pd.read_csv('../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv')
hybrid_new_df = pd.read_csv('../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv')

# Format the patient_id column as a 5-digit string and sort by patient_id
for df in [simple_avg_df, perf_weight_df, ttd_df, tta_df, hybrid_new_df]:
    df['patient_id'] = df['patient_id'].apply(lambda x: f"{int(x):05d}")
    df.sort_values('patient_id', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Create a combined DataFrame with the metrics of interest (including patient_id)
data = pd.DataFrame({
    'patient_id': simple_avg_df['patient_id'],
    'SimpleAvg_DiceOverall': simple_avg_df['Dice overall'],
    'PerfWeight_DiceOverall': perf_weight_df['Dice overall'],
    'TTD_DiceOverall': ttd_df['Dice overall'],
    # tta_df['Dice overall'],
    'Hybrid_DiceOverall': hybrid_new_df['Dice overall'],
    f'SimpleAvg_Dice{metric}': simple_avg_df[f'Dice {metric}'],
    f'PerfWeight_Dice{metric}': perf_weight_df[f'Dice {metric}'],
    f'TTD_Dice{metric}': ttd_df[f'Dice {metric}'],
    # f'TTA_Dice{metric}': tta_df[f'Dice {metric}'],
    f'Hybrid_Dice{metric}': hybrid_new_df[f'Dice {metric}'],
})

# --- 1. Find cases where one model is significantly better in Dice {metric} than the others ---

# Filter out cases where any model has Dice {metric} equal to 1 (likely indicating absence of the {metric} region)
filtered_data = data[(data[f'SimpleAvg_Dice{metric}'] < 1.0) & (data[f'PerfWeight_Dice{metric}'] < 1.0) &
                     (data[f'TTD_Dice{metric}'] < 1.0)  & (data[f'Hybrid_Dice{metric}'] < 1.0)]

def get_ordered_metrics(row, keys):
    # Create a dictionary of model names and their corresponding Dice {metric} values.
    values_dict = {k: row[k] for k in keys}
    # Sort model names based on their values.
    sorted_models = sorted(values_dict, key=lambda k: values_dict[k])
    best_model = sorted_models[-1]
    second_best_model = sorted_models[-2]
    best = values_dict[best_model]
    second_best = values_dict[second_best_model]
    diff = best - second_best
    return best, second_best, diff, best_model, second_best_model

keys = [f'SimpleAvg_Dice{metric}', f'PerfWeight_Dice{metric}', f'TTD_Dice{metric}', f'Hybrid_Dice{metric}' ]

# Apply row-wise: compute the best and second-best Dice {metric} values, their difference, and record the corresponding models.
filtered_data[[f'Best_Dice{metric}', f'SecondBest_Dice{metric}', f'Diff_Dice{metric}',
               f'Best_Model', f'Second_Best_Model']] = filtered_data.apply(
    lambda row: pd.Series(get_ordered_metrics(row, keys)), axis=1)

# Define a threshold for a significant difference (e.g., at least 0.10)
threshold_diff = 0.05
signif_cases = filtered_data[filtered_data[f'Diff_Dice{metric}'] >= threshold_diff]

print(f"Cases where one model significantly outperforms the others in Dice {metric}:")
print(signif_cases[['patient_id', f'Best_Dice{metric}', f'Best_Model', f'SecondBest_Dice{metric}', f'Second_Best_Model', f'Diff_Dice{metric}']])

# --- 2. Find cases where all models are inconsistent (high variance in Dice overall) ---

# Compute the variance of Dice overall across the four models for each patient.
data['DiceOverall_Var'] = data[['SimpleAvg_DiceOverall', 'PerfWeight_DiceOverall', 'TTD_DiceOverall', 'Hybrid_DiceOverall']].var(axis=1)

# Define inconsistency as those in the top 10th percentile of variance.
variance_threshold = data['DiceOverall_Var'].quantile(0.90)
inconsistent_cases = data[data['DiceOverall_Var'] >= variance_threshold]

print("\nCases with high inconsistency in Dice overall (top 10th percentile of variance):")
print(inconsistent_cases[['patient_id', 'SimpleAvg_DiceOverall', 'PerfWeight_DiceOverall', 'TTD_DiceOverall', 'Hybrid_DiceOverall', 'DiceOverall_Var']])