import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

metric = "ET"

# Define model display names mapping
model_display_names = {
    "simple_avg": "Simple Avg.",
    "perf_weight": "Perf. Weight",
    "ttd": "TTD",
    "tta": "TTA",
    "hybrid_new": "Hybrid"
}

correlations_path = f"../stats/results/spearman_correlations_features_Dice_{metric}_ensemble.csv"
correlations_df = pd.read_csv(correlations_path)

# Strip any leading/trailing whitespace from column names
correlations_df.columns = correlations_df.columns.str.strip()

# Define the desired order of models
model_order = ["simple_avg", "perf_weight", "ttd", "tta", "hybrid_new"]

# Pivot the data. If multiple rows exist for a given Model-Feature pair, aggregate by the mean.
heatmap_data = correlations_df.pivot_table(
    values="Spearman Correlation",
    index="Feature",
    columns="Model",
    aggfunc="mean"
)

# Filter model_order to only include models that exist in the data
available_models = [model for model in model_order if model in heatmap_data.columns]
heatmap_data = heatmap_data[available_models]

# Rename columns using display names
heatmap_data = heatmap_data.rename(columns=model_display_names)

# Create a larger figure
plt.figure(figsize=(24, 12))

# Increase the default font size for clarity
sns.set_context("notebook", font_scale=1.4)

# Create the heatmap
ax = sns.heatmap(
    heatmap_data, 
    annot=True, 
    cmap="coolwarm", 
    center=0, 
    linewidths=0.5, 
    fmt=".2f",
    # cbar_kws={"shrink": 0.8}  # optionally shrink the colorbar
)

# Rotate x-labels (model names) if needed
plt.xticks(rotation=45, ha="right")

# Rotate y-labels (features) if you prefer horizontal text
# otherwise, you can keep them as is. For clarity, let's keep them horizontal:
plt.yticks(rotation=0)

# Customize the plot
plt.title("Spearman Correlation Heatmap of MRI Features and Model Performance", fontsize=26, pad=20)
plt.xlabel("Model", fontsize=24, labelpad=10)
plt.ylabel("Feature", fontsize=24, labelpad=10)

# Ensure nothing is cut off
plt.tight_layout()

# Save the figure before showing
plt.savefig(f"./Figures/features_perf_heatmap_{metric}_ensemble.png", dpi=300, bbox_inches="tight")
plt.show()
