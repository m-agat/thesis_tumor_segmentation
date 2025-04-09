import pandas as pd
import scipy.stats as stats
import argparse
import os
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def check_normality(data, group_col, metric):
    """
    For each group in the grouping column, perform a Shapiro–Wilk test on the metric values.
    Returns a dictionary with groups and their p-values.
    """
    groups = data[group_col].unique()
    normality = {}
    for group in groups:
        values = data.loc[data[group_col] == group, metric].dropna()
        stat, p_value = stats.shapiro(values)
        normality[group] = p_value
        print(f"Group: {group:15s} | Shapiro-Wilk statistic: {stat:.4f}, p-value: {p_value:.4f}")
    return normality

def choose_test(data, group_col, metric, normality):
    """
    If all groups are normally distributed (p > 0.05), perform one-way ANOVA.
    Otherwise, perform the Kruskal-Wallis test.
    """
    groups = data[group_col].unique()
    samples = [data.loc[data[group_col] == group, metric].dropna() for group in groups]
    
    if all(p > 0.05 for p in normality.values()):
        print("\nAll groups appear to be normally distributed. Using parametric test: one-way ANOVA.")
        stat, p_value = stats.f_oneway(*samples)
        test_name = "ANOVA"
    else:
        print("\nAt least one group is not normally distributed. Using non-parametric test: Kruskal-Wallis.")
        stat, p_value = stats.kruskal(*samples)
        test_name = "Kruskal-Wallis"
    
    print(f"{test_name} test result: statistic = {stat:.4f}, p-value = {p_value:.4f}")
    return test_name, stat, p_value

def posthoc_test(data, group_col, metric, test_type):
    """
    Perform post-hoc pairwise comparisons based on the overall test type.
    For ANOVA, use Tukey's HSD.
    For non-parametric, use pairwise Mann-Whitney U tests with Bonferroni correction.
    """
    groups = data[group_col].unique()
    print("\nPost-hoc pairwise comparisons:")
    
    if test_type == "ANOVA":
        # Tukey's HSD test using statsmodels
        tukey = pairwise_tukeyhsd(endog=data[metric].dropna(), groups=data[group_col].dropna(), alpha=0.05)
        print(tukey.summary())
    else:
        # Non-parametric: pairwise Mann-Whitney U tests with Bonferroni correction.
        comparisons = list(itertools.combinations(groups, 2))
        p_values = []
        for g1, g2 in comparisons:
            sample1 = data.loc[data[group_col] == g1, metric].dropna()
            sample2 = data.loc[data[group_col] == g2, metric].dropna()
            stat, p = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            p_values.append(p)
            print(f"Comparison: {g1} vs {g2}: Mann-Whitney U p-value = {p:.4f}")
        # Apply Bonferroni correction
        bonferroni_p = [min(p * len(comparisons), 1.0) for p in p_values]
        print("\nAfter Bonferroni correction:")
        for (g1, g2), p_corr in zip(comparisons, bonferroni_p):
            print(f"Comparison: {g1} vs {g2}: corrected p-value = {p_corr:.4f}")

def main(csvs, models, metric):
    # Convert comma-separated arguments into lists
    csv_files = [s.strip() for s in csvs.split(',')]
    model_names = [s.strip() for s in models.split(',')]
    
    if len(csv_files) != len(model_names):
        raise ValueError("The number of CSV files must match the number of model names.")
    
    # Load and combine the data
    data = load_data(csv_files, model_names)
    print(f"Combined data shape: {data.shape}")
    
    # Check normality for each model group
    print(f"\nChecking normality for metric '{metric}' grouped by 'Model':")
    normality = check_normality(data, "Model", metric)
    
    # Choose and perform the appropriate test
    test_type, overall_stat, overall_p = choose_test(data, "Model", metric, normality)
    
    # Perform post-hoc pairwise comparisons
    posthoc_test(data, "Model", metric, test_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check distribution and choose statistical test across multiple CSV files.")
    parser.add_argument("--csvs", required=True,
                        help="Comma-separated list of CSV file paths (one per model).")
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of model names corresponding to the CSV files.")
    parser.add_argument("--metric", required=True,
                        help="The column name for the performance metric to test (e.g., 'Dice overall').")
    args = parser.parse_args()
    
    main(args.csvs, args.models, args.metric)


# python compare_models.py --csvs "../models/performance/vnet/patient_metrics_test.csv,../models/performance/segresnet/patient_metrics_test.csv,../models/performance/attunet/patient_metrics_test.csv,../models/performance/swinunetr/patient_metrics_test.csv" --models "VNet,SegResNet,AttUNet,SwinUNETR" --metric "Dice overall"
