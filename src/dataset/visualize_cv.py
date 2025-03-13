import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Example data: Replace this with your actual folds_data
json_path = "/home/magata/data/brats2021challenge/splits/data_splits.json"
with open(json_path, "r") as f:
    folds_data = json.load(f)

# Convert data to a DataFrame for easier manipulation
train_data = pd.DataFrame(folds_data["training"])
val_data = pd.DataFrame(folds_data["validation"])
test_data = pd.DataFrame(folds_data["test"])

# Aggregate the data to calculate counts
train_counts = (
    train_data.groupby(["fold", "region_combination"]).size().reset_index(name="count")
)
val_counts = (
    val_data.groupby(["fold", "region_combination"]).size().reset_index(name="count")
)

print(test_data.groupby(["region_combination"]).size().reset_index(name="count"))

# Pivot data for heatmap
heatmap_data = train_counts.pivot(
    index="region_combination", columns="fold", values="count"
).fillna(0)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt="g")
plt.title("Distribution of Region Combinations in Training Splits")
plt.xlabel("Fold")
plt.ylabel("Region Combination")
plt.savefig("regions_distribution_cv_heatmap.png")
plt.close()

# Plot bar charts for train, val, and test splits
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for ax, (name, data) in zip(
    axes, [("Train", train_data), ("Validation", val_data), ("Test", test_data)]
):
    sns.countplot(
        y="region_combination",
        data=data,
        ax=ax,
        order=data["region_combination"].value_counts().index,
    )
    ax.set_title(f"{name} Split Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Region Combination")

plt.tight_layout()
plt.savefig("regions_distribution_cv_barplot.png")
plt.close()
