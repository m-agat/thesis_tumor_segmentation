import pandas as pd 
import os 
from scipy.stats import spearmanr

# List of models for which performance data is available
models = ["vnet", "segresnet", "attunet", "swinunetr"]

# Load performance data for each model
performance_data = {
    model_name: pd.read_csv(f"../models/performance/{model_name}/patient_metrics_test_{model_name}.csv")
    for model_name in models
}

# Load the extracted features (from all modalities)
features = pd.read_csv("test_set_tumor_stats_all_modalities.csv")

# List of feature columns in the features file (excluding patient_id)
feature_columns = [col for col in features.columns if col != "patient_id"]

correlation_results = []
metric = "Dice ET"

for model_name in models:
    print("Model:", model_name)
    patient_metrics_df = performance_data[model_name]
    
    # Merge performance and features data on patient_id so that rows match.
    merged = pd.merge(patient_metrics_df, features, on="patient_id", how="inner")
    
    # Extract the Dice overall scores from the merged DataFrame
    scores = merged[metric]
    
    # Iterate over each feature column and compute the Spearman correlation.
    for feature in feature_columns:
        feat_values = merged[feature]
        correlation, p_val = spearmanr(feat_values, scores, nan_policy="omit")
        print(f"Feature: {feature}")
        print(f"Spearman Correlation: {correlation:.4f}, p-value: {p_val:.4g}\n")
        
        if p_val < 0.05:
            correlation_results.append({
                "Model": model_name,
                "Feature": feature,
                "Spearman Correlation": correlation,
                "p-value": p_val
            })

# Create a DataFrame of the correlation results
correlation_df = pd.DataFrame(correlation_results)
print(correlation_df)

# Save the results to a CSV file.
correlation_csv_path = f"spearman_correlations_features_{metric}.csv"
os.makedirs(os.path.dirname(correlation_csv_path) or ".", exist_ok=True)
correlation_df.to_csv(correlation_csv_path, index=False)
print(f"Correlations saved to: {correlation_csv_path}")
