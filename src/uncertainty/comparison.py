import pandas as pd
import numpy as np

# Load the two CSV files
file1 = "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/uncertainty/outputs/subregion_ensemble_performance_full.csv"
file2 = "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/subregion_swinunetr_performance_full.csv"

df1 = pd.read_csv(file1)
df1 = df1.replace([np.inf, -np.inf], np.nan)
df2 = pd.read_csv(file2)
df2 = df2.replace([np.inf, -np.inf], np.nan)

# Set a threshold for excluding large HD95 scores
hd95_threshold = 100  # Adjust this threshold as needed

# Filter rows with HD95 below the threshold
hd95_columns = [col for col in df1.columns if "HD95" in col]
df1_filtered = df1[(df1[hd95_columns] < hd95_threshold).all(axis=1)]
df2_filtered = df2[(df2[hd95_columns] < hd95_threshold).all(axis=1)]

# Ensure both files have a common identifier for filtering, e.g., 'Patient'
common_column = "Patient"

# Filter rows in the second DataFrame to match rows in the first
df2_filtered = df2_filtered[
    df2_filtered[common_column].isin(df1_filtered[common_column])
]

# Compute column-wise averages
average_df1 = df1_filtered.mean(
    numeric_only=True
)  # Compute average for all rows in the first file
average_df2 = df2_filtered.mean(
    numeric_only=True
)  # Compute average for filtered rows in the second file

# Compare the averages
comparison = pd.DataFrame(
    {
        "Ensemble Average": average_df1,
        "Swinunetr Filtered Average": average_df2,
        "Difference": average_df1 - average_df2,
    }
)

# Display the comparison
print(comparison)

# Save the comparison to a CSV file
comparison.to_csv(
    "/mnt/c/Users/agata/Desktop/thesis_tumor_segmentation/src/uncertainty/outputs/average_comparison.csv"
)
print("Comparison saved to 'average_comparison.csv'.")
