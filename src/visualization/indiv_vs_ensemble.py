import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(csv_files, model_names):
    """
    Loads each CSV file, adds a column 'Model' based on the provided model name,
    and concatenates all data into a single DataFrame.
    """
    data_frames = []
    for file, model in zip(csv_files, model_names):
        if not os.path.isfile(file):
            print(f"File not found: {file}")
            continue
        df = pd.read_csv(file)
        df['Model'] = model
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def add_significance_markers(ax, means, sems, significant_pairs, model_names):
    """
    Add significance markers between significantly different groups with automatic
    positioning and enough headroom to show them on large‐scale metrics like HD95.
    """
    if not significant_pairs:
        return

    # --- 1) compute a dynamic offset based on axis span ---
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    # choose, say, 5% of your span for the gap between levels,
    # and 2% of your span for the little "cap" on top of the bars
    base_offset = 0.05 * span
    text_height = 0.02 * span

    # prepare data structures
    used_positions = {i: [] for i in range(len(model_names))}

    # register your existing bar tops + error bars so you don't collide
    for i, (mean, sem) in enumerate(zip(means, sems)):
        top = mean + sem
        used_positions[i].append(top + text_height)

    # sort your pairs so wide comparisons go first (optional but helps readability)
    pairs = sorted(significant_pairs,
                   key=lambda pair: abs(model_names.index(pair[0]) - model_names.index(pair[1])),
                   reverse=True)

    for g1, g2 in pairs:
        i1, i2 = model_names.index(g1), model_names.index(g2)
        lo, hi = min(i1, i2), max(i1, i2)

        # find the highest “occupied” y within this span
        block_max = max(
            means[i] + sems[i] + text_height
            for i in range(lo, hi + 1)
        )

        # now slide up in increments of base_offset until you find a free slot
        level = block_max + base_offset
        conflict = True
        while conflict:
            conflict = False
            for x in range(lo, hi + 1):
                for used in used_positions[x]:
                    if abs(level - used) < (base_offset * 0.8):
                        conflict = True
                        break
                if conflict:
                    level += base_offset
                    break

        # draw your lines *without clipping*
        ax.plot([i1, i2], [level, level], 'k-', lw=1, clip_on=False)
        ax.plot([i1, i1], [level - (0.01 * span), level], 'k-', lw=1, clip_on=False)
        ax.plot([i2, i2], [level - (0.01 * span), level], 'k-', lw=1, clip_on=False)
        ax.text((i1 + i2) / 2, level + (0.005 * span), '*',
                ha='center', va='bottom', fontsize=12, clip_on=False)

        # remember you’ve used that height
        for x in range(lo, hi + 1):
            used_positions[x].append(level)

    # finally, extend your y‐axis so none of that is cut off
    new_max = max(max(pos_list) for pos_list in used_positions.values())
    ax.set_ylim(ymin, new_max + base_offset)


# Load significant results
# sig_results = pd.read_csv('../stats/significant_results_all_models.csv')
sig_results = pd.read_csv('../stats/significant_results_indiv.csv')

# Filter for HD95 metrics
sig_results = sig_results[sig_results['metric'].str.contains('HD95')]

csv_files = [
    # "../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv",
    # "../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv", 
    # "../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv",
    # "../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv", 
    # "../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv",
    "../models/performance/vnet/patient_metrics_test_vnet.csv",  
    "../models/performance/segresnet/patient_metrics_test_segresnet.csv", 
    "../models/performance/attunet/patient_metrics_test_attunet.csv", 
    "../models/performance/swinunetr/patient_metrics_test_swinunetr.csv" 
]
model_names = [
    # "Simple-Avg",
    # "Performance-Weighted",
    # "TTD",
    # "Hybrid",
    # "TTA",
    "VNet",
    "SegResNet",
    "AttUNet",
    "SwinUNETR"
]

data = load_data(csv_files, model_names)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define metrics to plot and their corresponding colors
metrics = ['HD95 NCR', 'HD95 ED', 'HD95 ET']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Plot bars for each metric
for i, (metric, color) in enumerate(zip(metrics, colors)):
    # Calculate mean and standard error for each model
    means = data.groupby('Model')[metric].mean()
    sems = data.groupby('Model')[metric].sem()

    means = means.reindex(model_names)
    sems  = sems.reindex(model_names)
    
    # Create bar plot
    ax = axes[i]
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=color)
    
    # Customize plot
    ax.set_title(metric, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('HD95 Score', fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar, mean, sem in zip(bars, means, sems):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', rotation=0, fontsize=12)
    
    # Add significance markers
    metric_sig_results = sig_results[sig_results['metric'] == metric]
    significant_pairs = list(zip(metric_sig_results['group1'], metric_sig_results['group2']))
    add_significance_markers(ax, means, sems, significant_pairs, model_names)

# Adjust layout
plt.tight_layout()
plt.savefig('./Figures/hd95_scores_barplots_indiv.png', dpi=300, bbox_inches='tight')
plt.show()
