import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set whether to plot model performance or ensemble performance.
# Choose plot_type = "model" or "ensemble"
plot_type = "model"  # change to "model" to plot model results

if plot_type == "model":
    base_path = r"..\models\performance"
    group_names = {"vnet": "V-Net", 
                   "segresnet": "SegResNet", 
                   "attunet": "Attention UNet", 
                   "swinunetr": "SwinUNETR"}
    file_pattern = "patient_metrics_test"
elif plot_type == "ensemble":
    base_path = r"..\ensemble\output_segmentations"
    group_names = {"simple_avg": "Simple Avg",
                   "perf_weight": "Performance Weighted",
                   "tta": "TTA-only", 
                   "ttd": "TTD-only", 
                   "hybrid_new": "Hybrid (TTD+TTA)"}
    file_pattern = "{}_patient_metrics_test.csv"
else:
    raise ValueError("plot_type must be 'model' or 'ensemble'.")

# Define Dice metrics to plot
dice_metrics = ["Dice NCR", "Dice ED", "Dice ET"]

# Set style parameters
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Function to create violin plot
def create_violin_plot(data, labels, metric, ax, title=None):
    sns.violinplot(data=pd.DataFrame({'Value': data, 'Group': labels}),
                  x='Group', y='Value', ax=ax,
                  inner='box',
                  cut=0)
    
    ax.set_title(title if title else metric, fontsize=20)
    ax.set_ylabel('Dice Score', fontsize=18)
    ax.set_xlabel('Ensemble Model', fontsize=18)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis="both", labelsize=15)

# Create the combined subplot figure
fig_combined, axs = plt.subplots(1, len(dice_metrics), figsize=(18, 6), sharey=True)

# Dictionary to store data for each metric
metric_data = {metric: {'values': [], 'labels': []} for metric in dice_metrics}

# Collect data from all groups
for key, label in group_names.items():
    if plot_type == "ensemble":
        csv_file = os.path.join(base_path, key, file_pattern.format(key))
    else:
        csv_file = os.path.join(base_path, key, f"{file_pattern}_{key}.csv")
    df = pd.read_csv(csv_file)
    
    for metric in dice_metrics:
        metric_data[metric]['values'].extend(df[metric].values)
        metric_data[metric]['labels'].extend([label] * len(df[metric]))

# Create combined subplot figure
for i, metric in enumerate(dice_metrics):
    create_violin_plot(
        metric_data[metric]['values'],
        metric_data[metric]['labels'],
        metric,
        axs[i]
    )
    if i != 0:  # Remove y-label for all but first subplot
        axs[i].set_ylabel('')

plt.suptitle("Patient-Level Distribution of Dice Metrics", fontsize=20, y=1.05)
plt.tight_layout()
plt.savefig("./Figures/violin_plots_dice_metrics_combined_indiv.png", bbox_inches='tight', dpi=300)

# Create individual figures for each metric
for metric in dice_metrics:
    fig_individual = plt.figure(figsize=(8, 6))
    ax = fig_individual.add_subplot(111)
    
    create_violin_plot(
        metric_data[metric]['values'],
        metric_data[metric]['labels'],
        metric,
        ax,
        title=f"Patient-Level Distribution of {metric}"
    )
    
    plt.tight_layout()
    # Save individual plot with metric name in filename
    metric_filename = metric.replace(" ", "_").lower()
    plt.savefig(f"./Figures/violin_plot_{metric_filename}.png", bbox_inches='tight', dpi=300)
    plt.close()

# Show the combined plot
plt.show()