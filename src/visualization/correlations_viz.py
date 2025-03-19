import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

correlations_path = "../uncertainty/outputs/correlations/spearman_correlations_composite_score.csv"
correlations_df = pd.read_csv(correlations_path)

# Pivot the data to create a heatmap-friendly format
heatmap_data = correlations_df.pivot_table(
    values="Spearman Correlation",
    index=["Model"],
    columns=["Uncertainty Method", "Subregion"]
)

# Create the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", center=0, linewidths=0.5, fmt=".2f")

# Customize the plot
plt.title("Spearman Correlation Heatmap of Model Uncertainty and Performance")
plt.xlabel("Uncertainty Method & Subregion")
plt.ylabel("Model")

# Display the plot
plt.show()
