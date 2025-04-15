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

def get_numeric_metrics(data):
    """
    Returns a list of numeric metric columns from the DataFrame.
    Excludes non-numeric columns like 'Model' and 'Patient'.
    """
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    return [col for col in numeric_cols if col not in ['Model', 'Patient']]

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
    Returns a list of significant comparisons with their p-values.
    """
    groups = data[group_col].unique()
    significant_comparisons = []
    
    if test_type == "ANOVA":
        tukey = pairwise_tukeyhsd(endog=data[metric].dropna(), groups=data[group_col].dropna(), alpha=0.05)
        results = tukey.summary().data[1:]  # Skip header row
        for row in results:
            if float(row[4]) < 0.05:  # p-value column
                significant_comparisons.append({
                    'group1': row[0],
                    'group2': row[1],
                    'p_value': float(row[4])
                })
    else:
        comparisons = list(itertools.combinations(groups, 2))
        for g1, g2 in comparisons:
            sample1 = data.loc[data[group_col] == g1, metric].dropna()
            sample2 = data.loc[data[group_col] == g2, metric].dropna()
            stat, p = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            p_corrected = min(p * len(comparisons), 1.0)  # Bonferroni correction
            if p_corrected < 0.05:
                significant_comparisons.append({
                    'group1': g1,
                    'group2': g2,
                    'p_value': p_corrected
                })
    
    return significant_comparisons

def main(csvs, models, output_file):
    # Convert comma-separated arguments into lists
    csv_files = [s.strip() for s in csvs.split(',')]
    model_names = [s.strip() for s in models.split(',')]
    
    if len(csv_files) != len(model_names):
        raise ValueError("The number of CSV files must match the number of model names.")
    
    # Load and combine the data
    data = load_data(csv_files, model_names)
    print(f"Combined data shape: {data.shape}")
    
    # Get all numeric metrics
    metrics = get_numeric_metrics(data)
    print(f"Found {len(metrics)} metrics to analyze")
    
    # Store significant results
    significant_results = []
    
    # Analyze each metric
    for metric in metrics:
        print(f"\nAnalyzing metric: {metric}")
        
        # Check normality
        normality = check_normality(data, "Model", metric)
        
        # Choose and perform the appropriate test
        test_type, overall_stat, overall_p = choose_test(data, "Model", metric, normality)
        
        # If overall test is significant, perform post-hoc tests
        if overall_p < 0.05:
            print(f"Significant overall difference found for {metric} (p = {overall_p:.4f})")
            posthoc_comparisons = posthoc_test(data, "Model", metric, test_type)
            
            # Add significant results to the list
            for comp in posthoc_comparisons:
                significant_results.append({
                    'metric': metric,
                    'test_type': test_type,
                    'overall_p_value': overall_p,
                    'group1': comp['group1'],
                    'group2': comp['group2'],
                    'posthoc_p_value': comp['p_value']
                })
    
    # Save significant results to CSV
    if significant_results:
        results_df = pd.DataFrame(significant_results)
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved {len(significant_results)} significant results to {output_file}")
    else:
        print("\nNo significant differences found for any metric.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform statistical tests across multiple CSV files for all metrics.")
    parser.add_argument("--csvs", required=True,
                        help="Comma-separated list of CSV file paths (one per model).")
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of model names corresponding to the CSV files.")
    parser.add_argument("--output", required=True,
                        help="Output CSV file path to save significant results.")
    args = parser.parse_args()
    
    main(args.csvs, args.models, args.output)

# Example usage:
# python compare_models.py --csvs "model1.csv,model2.csv,model3.csv" --models "Model1,Model2,Model3" --output "significant_results.csv"

# python compare_models.py --csvs "../models/performance/vnet/patient_metrics_test.csv,../models/performance/segresnet/patient_metrics_test.csv,../models/performance/attunet/patient_metrics_test.csv,../models/performance/swinunetr/patient_metrics_test.csv" --models "VNet,SegResNet,AttUNet,SwinUNETR" --output "significant_results.csv"

# python compare_models.py --csvs "../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv,../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv, ../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv,../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv, ../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv" --models "Simple Avg,Performance Weighted,TTD,Hybrid,TTA" --output "significant_results.csv"

# python compare_models.py --csvs "../models/performance/segresnet/patient_metrics_test_segresnet.csv,../models/performance/attunet/patient_metrics_test_attunet.csv,../models/performance/swinunetr/patient_metrics_test_swinunetr.csv,../ensemble/output_segmentations/hybrid_new/hybrid_patient_metrics_test.csv" --models "SegResNet,Attention UNet,SwinUNETR,Hybrid" --output "significant_results.csv"
