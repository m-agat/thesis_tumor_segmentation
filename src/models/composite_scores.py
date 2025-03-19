import pandas as pd 

model_name = "swinunetr"
metrics_path = f"./performance/{model_name}/patient_metrics.csv"
metrics_df = pd.read_csv(metrics_path)

# Define composite score weights
composite_score_weights = {
    "Dice": 0.45,
    "HD95": 0.15,
    "Sensitivity": 0.3,
    "Specificity": 0.1,
}

# Function to compute composite scores
def compute_composite_scores(metrics, weights):
    """Compute weighted composite scores for a model."""
    composite_scores = {}
    
    # Background composite score
    normalized_hd95_bg = 1 / (1 + metrics["HD95 BG"])
    composite_scores["BG"] = (
        weights["Dice"] * metrics["Dice BG"]
        + weights["HD95"] * normalized_hd95_bg
        + weights["Sensitivity"] * metrics["Sensitivity BG"]
        + weights["Specificity"] * metrics["Specificity BG"]
    )
    
    # Tumor regions composite scores
    for region in ["NCR", "ED", "ET"]:
        normalized_hd95 = 1 / (1 + metrics[f"HD95 {region}"])
        composite_scores[region] = (
            weights["Dice"] * metrics[f"Dice {region}"]
            + weights["HD95"] * normalized_hd95
            + weights["Sensitivity"] * metrics[f"Sensitivity {region}"]
            + weights["Specificity"] * metrics[f"Specificity {region}"]
        )
    
    return composite_scores

# Compute composite scores for all patients
composite_scores_list = []
for _, row in metrics_df.iterrows():
    patient_id = row["patient_id"]
    patient_metrics = row.to_dict()
    composite_scores = compute_composite_scores(patient_metrics, composite_score_weights)
    composite_scores["patient_id"] = patient_id
    composite_scores_list.append(composite_scores)

# Convert to DataFrame
composite_scores_df = pd.DataFrame(composite_scores_list)
composite_scores_df["patient_id"] = metrics_df["patient_id"].astype(str).str.zfill(5)

# Reorder columns
column_order = ["patient_id", "BG", "NCR", "ED", "ET"]
composite_scores_df = composite_scores_df[column_order]

# Save to CSV
composite_scores_csv_path = f"./performance/{model_name}/composite_scores.csv"
composite_scores_df.to_csv(composite_scores_csv_path, index=False)
