import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def convert_tuple_to_mean(value):
    if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
        try:
            return np.mean(eval(value))
        except:
            return np.nan
    return value

# Load data
model = "vnet"
features_df = pd.read_csv("./outputs/mri_features.csv")
performance_df = pd.read_csv(f"/home/magata/results/metrics/patient_performance_scores_{model}.csv")

# Merge features and performance metrics
merged_df = pd.merge(features_df, performance_df, on='Patient')
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Define the target column
target_col_name = 'Composite_Score_4'

# Select only MRI feature columns (exclude Patient and performance metrics)
# Assuming performance-related columns in performance_df other than 'Patient' and 'Dice_1' start with "Dice" or are known
performance_columns = [col for col in performance_df.columns if col not in ['Patient', target_col_name]]
columns_to_exclude = ['Patient'] + performance_columns

features = features_df.select_dtypes(include=[np.number])

target = merged_df[target_col_name]  # Select the target column

# # Compute correlation
correlations = features.corrwith(target)

# Display correlations
print(f"Correlations between MRI features and {target_col_name}:")
print(correlations)

# Save correlations to a CSV
correlations_df = correlations.reset_index()  # Convert to DataFrame for saving
correlations_df.columns = ['Feature', 'Correlation']  # Add meaningful column names
correlations_df.to_csv(f'./outputs/correlations_all_features_{target_col_name}_{model}.csv', index=False)

# Visualize correlations as a bar plot
correlations.plot(kind='bar', title=f'Feature Correlation with {target_col_name}', figsize=(10, 6))
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.savefig(f"./outputs/correlations_{target_col_name}_{model}.png")

print("Correlations and plot saved to './outputs/'.")