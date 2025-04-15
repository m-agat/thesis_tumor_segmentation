import os
import json
import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Data Loading (Example)
# -------------------
model_names = ["VNet", "SegResNet", "AttUNet", "SwinUNETR"]
base_path = r"..\models\performance"

# Define the metrics to include in the radar chart.
# We'll invert HD95 so that "higher is better" (Dice-like interpretation).
metrics = ["HD95 (inv)", "Dice", "Sensitivity", "Specificity"]

def load_model_metrics(model):
    """
    Load and return a list of metric values in the order defined by 'metrics', 
    computed as the average over non-background values.
    """
    avg_file = os.path.join(base_path, model.lower(), "average_metrics_test.json")
    with open(avg_file, 'r') as f:
        data = json.load(f)
    
    # Compute overall scores excluding background by averaging the tumor sub-region metrics.
    dice_overall = (data["Dice NCR"] + data["Dice ED"] + data["Dice ET"]) / 3
    hd95_overall = data["HD95 overall"]
    sensitivity_overall = (data["Sensitivity NCR"] + data["Sensitivity ED"] + data["Sensitivity ET"]) / 3
    specificity_overall = (data["Specificity NCR"] + data["Specificity ED"] + data["Specificity ET"]) / 3

    values = []
    for m in metrics:
        if m == "HD95 (inv)":
            # Invert the overall HD95 value so that higher values indicate better performance.
            values.append(1 / hd95_overall)
        elif m == "Dice":
            values.append(dice_overall)
        elif m == "Sensitivity":
            values.append(sensitivity_overall)
        elif m == "Specificity":
            values.append(specificity_overall)
    return values

# Gather metric values for each model.
model_values = {m: load_model_metrics(m) for m in model_names}

# -------------------
# Radar Chart Setup
# -------------------
N = len(metrics)  # Number of metrics
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # Close the loop

# Increase figure size
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Place the first metric at the top and rotate clockwise.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Define a color palette for the models.
colors = ["#ff9f7f", "#ffc154", "#ffa8df", "#c689ff"]

# Plot each model.
for idx, model in enumerate(model_names):
    values = model_values[model]
    values = np.concatenate((values, [values[0]]))  # Close the loop.
    
    ax.plot(angles, values, color=colors[idx], linewidth=2, label=model)
    ax.fill(angles, values, color=colors[idx], alpha=0.3)

# -------------------
# Adding the Scale (Radial Ticks)
# -------------------
radial_ticks = np.linspace(0, 1, 5)
ax.set_yticks(radial_ticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in radial_ticks], fontsize=12)
ax.set_rlabel_position(135)  # Position the radial labels

# -------------------
# Styling the Radar Chart
# -------------------
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=13)

ax.set_rmax(1.0)
ax.set_rmin(0.0)

ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.7)
ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.7)

ax.spines['polar'].set_visible(False)

# -------------------
# Title and Legend
# -------------------
ax.set_title("Comparison of Model Configurations (Excluding Background)", y=1.08, fontsize=16)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

plt.tight_layout()
plt.savefig("./Figures/radar_chart_with_scale.png")
plt.show()
