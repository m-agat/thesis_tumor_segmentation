import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

model_names = {"vnet": "V-Net", "segresnet": "SegResNet", "attunet": "Attention UNet", "swinunetr": "SwinUNETR"}
base_path = r"..\models\performance"

# Define the sub-regions and the associated keys for Dice scores
sub_regions = ["NCR", "ED", "ET"]
dice_keys = [f"Dice {sub}" for sub in sub_regions]

# Define a color mapping for the sub-regions
region_colors = {
    "NCR": "#e74c3c",  
    "ED": "#f1c40f",   
    "ET": "#3498db"
}

# Initialize a dictionary to store average dice metrics per model
model_metrics = {}

for model in model_names.keys():
    avg_file = os.path.join(base_path, model, "average_metrics_test.json")
    with open(avg_file, 'r') as f:
        data = json.load(f)
    model_metrics[model] = {key: data[key] for key in dice_keys}

# Dictionaries to store mean values and std values per model for each sub-region
model_stds = {key: [] for key in dice_keys}

# Loop through each model to load CSVs and compute metrics
for model in model_names.keys():
    csv_file = os.path.join(base_path, model, "patient_metrics_test.csv")
    df = pd.read_csv(csv_file)
    for key in dice_keys:
        model_stds[key].append(df[key].std())

# Prepare data for plotting.
# For each sub-region, gather dice scores for all models.
values = {key: [model_metrics[model][key] for model in model_names.keys()] for key in dice_keys}

x = np.arange(len(model_names))  # positions for each model
width = 0.2  # width of each bar

fig, ax = plt.subplots(figsize=(14, 8))
for i, key in enumerate(dice_keys):
    # Extract sub-region from the key, e.g., "Dice NCR" -> "NCR"
    region = key.split()[-1]
    color = region_colors.get(region, None)
    ax.bar(x + i * width, values[key], width, yerr=model_stds[key], label=region, color=color, capsize=5)

ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Dice Score', fontsize=14)
ax.set_title('Dice Scores by Sub-Region for Each Model', fontsize=16)
ax.set_xticks(x + width * (len(dice_keys) - 1) / 2)
ax.set_xticklabels(model_names.values(), fontsize=12)
ax.legend(title='Sub-Region', loc="upper left", fontsize=12, title_fontsize=13)
plt.tight_layout()
plt.savefig("./Figures/dice_scores_all_models.png")
plt.show()