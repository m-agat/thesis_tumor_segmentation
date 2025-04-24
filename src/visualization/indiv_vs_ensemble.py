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
    Add significance markers between significantly different groups with smart positioning
    to avoid overlaps and ensure they're above value labels
    """
    if not significant_pairs:
        return
        
    # Sort pairs by distance between bars
    sorted_pairs = []
    for pair in significant_pairs:
        group1, group2 = pair
        idx1 = model_names.index(group1)
        idx2 = model_names.index(group2)
        distance = abs(idx2 - idx1)
        sorted_pairs.append((distance, group1, group2))
    
    sorted_pairs.sort(reverse=True)  # Sort by distance, longest first
    
    # Keep track of used y-positions for each x-position
    used_positions = {i: [] for i in range(len(model_names))}
    base_offset = 0.05  # Increased base spacing to account for value labels
    text_height = 0.02  # Height needed for value text
    
    # First, add all value positions to used_positions
    for i, (mean, sem) in enumerate(zip(means, sems)):
        height = mean + sem + text_height
        used_positions[i].append(height)
    
    for _, group1, group2 in sorted_pairs:
        idx1 = model_names.index(group1)
        idx2 = model_names.index(group2)
        start_idx = min(idx1, idx2)
        end_idx = max(idx1, idx2)
        
        # Find the maximum height including error bars and text
        max_height = max(means[i] + sems[i] + text_height 
                        for i in range(start_idx, end_idx + 1))
        
        # Find the minimum available y-position that doesn't overlap
        y_offset = base_offset
        while True:
            position_taken = False
            test_y = max_height + y_offset
            
            # Check if this y-position is already used by any x-position in the range
            for x in range(start_idx, end_idx + 1):
                for used_y in used_positions[x]:
                    if abs(test_y - used_y) < base_offset:
                        position_taken = True
                        break
                if position_taken:
                    break
            
            if not position_taken:
                break
            y_offset += base_offset
        
        # Add the line at the chosen height
        line_height = max_height + y_offset
        
        # Add horizontal line
        ax.plot([idx1, idx2], [line_height] * 2, 'k-', lw=1)
        
        # Add vertical lines
        ax.plot([idx1, idx1], [line_height - 0.01, line_height], 'k-', lw=1)
        ax.plot([idx2, idx2], [line_height - 0.01, line_height], 'k-', lw=1)
        
        # Add asterisk
        ax.text((idx1 + idx2) / 2, line_height + 0.005, '*', 
                ha='center', va='bottom', fontsize=10)
        
        # Record the used y-positions
        for x in range(start_idx, end_idx + 1):
            used_positions[x].append(line_height)
            
    # Adjust the y-axis limit to accommodate all significance bars
    max_y = max(max(positions) for positions in used_positions.values() if positions)
    ax.set_ylim(0, max_y + 0.05)  # Add some padding above the highest bar

# Load significant results
sig_results = pd.read_csv('../stats/significant_results_all_models.csv')

# Filter for Speci metrics
sig_results = sig_results[sig_results['metric'].str.contains('Speci')]

csv_files = [
    "../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv",
    "../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv", 
    "../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv",
    "../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv", 
    "../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv", 
    "../models/performance/segresnet/patient_metrics_test_segresnet.csv", 
    "../models/performance/attunet/patient_metrics_test_attunet.csv", 
    "../models/performance/swinunetr/patient_metrics_test_swinunetr.csv" 
]
model_names = [
    "Simple-Avg",
    "Performance-Weighted",
    "TTD",
    "Hybrid",
    "TTA",
    "SegResNet",
    "AttUNet",
    "SwinUNETR"
]

data = load_data(csv_files, model_names)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define metrics to plot and their corresponding colors
metrics = ['Specificity NCR', 'Specificity ED', 'Specificity ET']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Plot bars for each metric
for i, (metric, color) in enumerate(zip(metrics, colors)):
    # Calculate mean and standard error for each model
    means = data.groupby('Model')[metric].mean()
    sems = data.groupby('Model')[metric].sem()
    
    # Create bar plot
    ax = axes[i]
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=sems, capsize=5, color=color)
    
    # Customize plot
    ax.set_title(metric, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Speci Score', fontsize=12)
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
plt.savefig('./Figures/specificity_scores_indiv_vs_ensemble.png', dpi=300, bbox_inches='tight')
plt.show()
