import pandas as pd 
import os 
from scipy.stats import spearmanr

# List of individual and ensemble models for which performance data is available
# individual_models = ["vnet", "segresnet", "attunet", "swinunetr"]
ensemble_models = ["simple_avg", "perf_weight", "ttd", "tta", "hybrid_new"]  # Add your ensemble model names here
models = ensemble_models

# Load performance data for each model
performance_data = {}
for model_name in ensemble_models:
    try:
        performance_data[model_name] = pd.read_csv(f"../ensemble/output_segmentations/{model_name}/{model_name}_patient_metrics_test.csv")
    except FileNotFoundError:
        print(f"Warning: Performance data not found for model {model_name}")
        continue

# Load the extracted features (from all modalities)
features = pd.read_csv("./results/test_set_tumor_stats_all_modalities.csv")

# List of feature columns in the features file (excluding patient_id)
feature_columns = [col for col in features.columns if col != "patient_id"]

correlation_results = []
metric = "Dice ET"

for model_name in ensemble_models:
    if model_name not in performance_data:
        continue
        
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
correlation_csv_path = f"./results/spearman_correlations_features_{metric}_ensemble.csv"
os.makedirs(os.path.dirname(correlation_csv_path) or ".", exist_ok=True)
correlation_df.to_csv(correlation_csv_path, index=False)
print(f"Correlations saved to: {correlation_csv_path}")
