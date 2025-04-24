import pandas as pd

# tta_df = pd.read_csv("../ensemble/output_segmentations/tta/ece_results_per_subregion.csv")
ttd_df = pd.read_csv("../ensemble/output_segmentations/ttd/ece_results_per_subregion.csv")
# hybrid_df = pd.read_csv("../ensemble/output_segmentations/hybrid_new/ece_results_per_subregion_hybrid.csv")
tta_df = pd.read_csv("../ensemble/output/tta/ece_results_per_subregion_tta.csv")# 
hybrid_df = pd.read_csv("../ensemble/output/hybrid/ece_results_per_subregion_hybrid1.csv") 

# Compute the average ECE for each subregion (NCR, ED, ET)
# Calculate mean for each subregion column
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for plotting
models = ['TTA', 'TTD', 'Hybrid']
regions = ['NCR', 'ED', 'ET']

data = {
    'NCR': [tta_df['NCR_ECE'], ttd_df['NCR_ECE'], hybrid_df['NCR_ECE']],
    'ED': [tta_df['ED_ECE'], ttd_df['ED_ECE'], hybrid_df['ED_ECE']],
    'ET': [tta_df['ET_ECE'], ttd_df['ET_ECE'], hybrid_df['ET_ECE']]
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Boxplot
bp_positions = np.arange(len(regions))
box_width = 0.25

for i, model in enumerate(models):
    positions = bp_positions + i*box_width - box_width
    boxes = [data[region][i] for region in regions]
    ax1.boxplot(boxes, positions=positions, widths=0.2, 
                labels=['']*len(regions), patch_artist=True)

ax1.set_xlabel('Tumor Subregions')
ax1.set_ylabel('Expected Calibration Error (ECE)')
ax1.set_title('ECE Distribution per Model and Subregion')
ax1.set_xticks(bp_positions)
ax1.set_xticklabels(regions)
ax1.legend(models, loc='upper right')

# Plot 2: Bar plot with error bars
bar_positions = np.arange(len(regions))
width = 0.25

for i, model in enumerate(models):
    means = [data[region][i].mean() for region in regions]
    stds = [data[region][i].std() for region in regions]
    ax2.bar(bar_positions + i*width, means, width, label=model, yerr=stds, capsize=5)

ax2.set_xlabel('Tumor Subregions')
ax2.set_ylabel('Expected Calibration Error (ECE)')
ax2.set_title('Average ECE with Standard Deviation')
ax2.set_xticks(bar_positions + width)
ax2.set_xticklabels(regions)
ax2.legend()

plt.tight_layout()
# plt.savefig('../visualization/ece_comparison.png', bbox_inches='tight', dpi=300)
plt.show()
plt.close()


for df, model_name in zip([tta_df, ttd_df, hybrid_df], ["tta", "ttd", "hybrid"]):
    avg_ece_subregions = {
        'NCR': df['NCR_ECE'].mean(), 
        'ED': df['ED_ECE'].mean(),
        'ET': df['ET_ECE'].mean()
    }
    std_ece_subregions = {
        'NCR': df['NCR_ECE'].std(), 
        'ED': df['ED_ECE'].std(),
        'ET': df['ET_ECE'].std()
    }
    print(model_name)
    print(avg_ece_subregions)
    print(std_ece_subregions)
    print("\n")

