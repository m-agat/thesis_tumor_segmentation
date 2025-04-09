import os
import pandas as pd
import matplotlib.pyplot as plt

# List of models and base folder for performance CSV files.
model_names = {"vnet": "V-Net", "segresnet": "SegResNet", "attunet": "Attention UNet", "swinunetr": "SwinUNETR"}
base_path = r"..\models\performance"

# Define which Dice metrics (sub-regions) to plot
dice_metrics = ["Dice NCR", "Dice ED", "Dice ET"]

# Create a figure with one subplot per metric
fig, axs = plt.subplots(1, len(dice_metrics), figsize=(18, 6), sharey=True)

for i, metric in enumerate(dice_metrics):
    data = []
    for model in model_names.keys():
        csv_file = os.path.join(base_path, model, "patient_metrics_test.csv")
        df = pd.read_csv(csv_file)
        data.append(df[metric])
    axs[i].boxplot(data, labels=model_names.values())
    axs[i].set_title(metric, fontsize=14)
    axs[i].set_ylabel(metric, fontsize=12)
    axs[i].tick_params(axis="both", labelsize=10)

plt.suptitle("Patient-Level Distribution of Dice Metrics Across Models", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./Figures/box_plots_dice_metrics.png")
plt.show()
