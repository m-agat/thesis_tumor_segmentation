import pandas as pd 
import numpy as np

# Average model results
model_results = {
    "AttentionUNet": pd.read_csv("/home/magata/results/metrics/patient_performance_scores_attunet.csv").T[1:].replace([np.inf, -np.inf], np.nan).mean(axis=1),
    "SegResNet": pd.read_csv("/home/magata/results/metrics/patient_performance_scores_segresnet.csv").T[1:].replace([np.inf, -np.inf], np.nan).mean(axis=1),
    "SwinUNetr": pd.read_csv("/home/magata/results/metrics/patient_performance_scores_swinunetr.csv").T[1:].replace([np.inf, -np.inf], np.nan).mean(axis=1),
    "VNet": pd.read_csv("/home/magata/results/metrics/patient_performance_scores_vnet.csv").T[1:].replace([np.inf, -np.inf], np.nan).mean(axis=1)
}

dfs = [pd.DataFrame(data, columns=[model_name]) for model_name, data in model_results.items()]

# Concatenate all DataFrames along columns
final_df = pd.concat(dfs, axis=1).T

# Save to CSV
# output_path = "/home/magata/results/metrics/model_performance_summary.csv"
output_path = "/mnt/c/Users/agata/OneDrive/Pulpit/thesis_tumor_segmentation/results/metrics/model_performance_summary.csv"
final_df.to_csv(output_path, index=True)
print(f"Results saved to {output_path}")

# class_weights = pd.read_csv("/home/magata/results/metrics/model_performance_summary.csv", index_col=0).to_dict(orient="index")
# print(class_weights)