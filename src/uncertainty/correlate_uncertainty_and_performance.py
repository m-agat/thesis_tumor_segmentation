import pandas as pd
from scipy.stats import spearmanr
import os

models = ["attunet", "segresnet", "swinunetr", "vnet"]
subregions = ["NCR", "ED", "ET"]
uncertainties = ["TTA", "TTD", "Hybrid"]
patient_uncertainties_path = "./outputs/uncertainties/patient_uncertainties.csv"

patient_uncertainties = pd.read_csv(patient_uncertainties_path)

performance_data = {
    model_name: pd.read_csv(f"../models/performance/{model_name}/patient_metrics.csv")
    for model_name in models
}

correlation_results = []

for model_name in models:
    patient_metrics_df = performance_data[model_name]

    for uncertainty_method in uncertainties:
        for subregion in subregions:
            uncertainty_vals = patient_uncertainties.loc[
                patient_uncertainties["Model"] == model_name,
                f"{uncertainty_method} {subregion}",
            ]
            dice_scores = patient_metrics_df[f"Dice {subregion}"]

            correlation, _ = spearmanr(uncertainty_vals, dice_scores)

            correlation_results.append(
                {
                    "Model": model_name,
                    "Uncertainty Method": uncertainty_method,
                    "Subregion": subregion,
                    "Spearman Correlation": correlation,
                }
            )

correlation_df = pd.DataFrame(correlation_results)
correlation_csv_path = "./outputs/correlations/spearman_correlations.csv"
os.makedirs(os.path.dirname(correlation_csv_path), exist_ok=True)
correlation_df.to_csv(correlation_csv_path, index=False)

print(f"Correlations saved to: {correlation_csv_path}")
