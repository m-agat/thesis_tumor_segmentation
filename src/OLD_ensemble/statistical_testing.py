import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import seaborn as sns 
import matplotlib.pyplot as plt 

# Load data
performance_df_swinunetr = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_swinunetr.csv")
performance_df_segnet = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_segresnet.csv")

t_stat, p_value = ttest_rel(performance_df_swinunetr['Dice_1'], performance_df_segnet['Dice_1'])
print(f"T-Test for NCR: t-statistic = {t_stat}, p-value = {p_value}")

# Wilcoxon signed-rank test for NCR
stat, p = wilcoxon(performance_df_swinunetr['Dice_1'], performance_df_segnet['Dice_1'])
print(f"Wilcoxon Test for NCR: statistic = {stat}, p-value = {p}")

t_stat, p_value = ttest_rel(performance_df_swinunetr['Dice_2'], performance_df_segnet['Dice_2'])
print(f"T-Test for ED: t-statistic = {t_stat}, p-value = {p_value}")

# Wilcoxon signed-rank test for NCR
stat, p = wilcoxon(performance_df_swinunetr['Dice_2'], performance_df_segnet['Dice_2'])
print(f"Wilcoxon Test for ED: statistic = {stat}, p-value = {p}")

# Paired t-test for NCR
t_stat, p_value = ttest_rel(performance_df_swinunetr['Dice_4'], performance_df_segnet['Dice_4'])
print(f"T-Test for ET: t-statistic = {t_stat}, p-value = {p_value}")

# Wilcoxon signed-rank test for NCR
stat, p = wilcoxon(performance_df_swinunetr['Dice_4'], performance_df_segnet['Dice_4'])
print(f"Wilcoxon Test for ET: statistic = {stat}, p-value = {p}")

# Calculate the performance difference
performance_diff_ncr = performance_df_swinunetr['Dice_1'] - performance_df_segnet['Dice_1']

# Create a DataFrame with the performance differences and a column for tissue type
performance_diff_df = pd.DataFrame({
    'Performance Difference': performance_diff_ncr,
    'Tissue': 'NCR'  # Assign tissue name as 'NCR' for this data
})

# Display a summary of the performance differences
print(performance_diff_ncr.describe())

# Plot the boxplot
# sns.boxplot(x='Tissue', y='Performance Difference', data=performance_diff_df)
# plt.title("Performance Difference between SwinUNETR and SegResNet for NCR")
# plt.savefig("./outputs/performance_diff_segnet_swinunetr.png")
# plt.show()