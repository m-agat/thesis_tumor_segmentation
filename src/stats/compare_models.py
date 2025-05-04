import argparse
import pandas as pd
from stats.data import load_data, get_numeric_metrics
from stats.stats_tests import (
    check_normality, choose_test,
    posthoc_anova, posthoc_nonparametric
)
from stats.utils import is_ensemble_model, format_relationship

def main(csvs: str, models: str, output: str):
    files = [p.strip() for p in csvs.split(",")]
    names = [n.strip() for n in models.split(",")]
    df = load_data(files, names)
    metrics = get_numeric_metrics(df)

    results = []
    for m in metrics:
        print(f"\n▶ Metric: {m}")
        norm = check_normality(df, "Model", m)
        test_name, stat, p = choose_test(df, "Model", m, norm)
        if p < 0.05:
            if test_name == "ANOVA":
                post = posthoc_anova(df, "Model", m)
            else:
                post = posthoc_nonparametric(df, "Model", m)
            for g1, g2, s, pp in post:
                # skip comparisons neither side is an ensemble?
                results.append({
                    "metric": m, "test": test_name,
                    "overall_stat": stat, "overall_p": p,
                    "group1": g1, "group2": g2,
                    "post_stat": s, "post_p": pp,
                    "relationship": format_relationship(
                        g1, ">" if s > 0 else "<", g2
                    )
                })

    if results:
        pd.DataFrame(results).to_csv(output, index=False)
        print(f"\nSaved {len(results)} significant comparisons → {output}")
    else:
        print("\nNo significant differences found.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csvs",   required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.csvs, args.models, args.output)

# Example usage:
# python compare_models.py --csvs "model1.csv,model2.csv,model3.csv" --models "Model1,Model2,Model3" --output "significant_results.csv"

# python3 compare_models.py --csvs "../models/performance/vnet/patient_metrics_test_vnet.csv,../models/performance/segresnet/patient_metrics_test_segresnet.csv,../models/performance/attunet/patient_metrics_test_attunet.csv,../models/performance/swinunetr/patient_metrics_test_swinunetr.csv" --models "VNet,SegResNet,AttUNet,SwinUNETR" --output "significant_results_indiv.csv"

# python compare_models.py --csvs "../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv,../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv, ../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv,../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv, ../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv, ../models/performance/segresnet/patient_metrics_test_segresnet.csv, ../models/performance/attunet/patient_metrics_test_attunet.csv, ../models/performance/swinunetr/patient_metrics_test_swinunetr.csv" --models "Simple-Avg,Performance-Weighted,TTD,Hybrid,TTA,SegResNet,AttUNet,SwinUNETR" --output "significant_results_all_models.csv"

# python compare_models.py --csvs "../models/performance/segresnet/patient_metrics_test_segresnet.csv,../models/performance/attunet/patient_metrics_test_attunet.csv,../models/performance/swinunetr/patient_metrics_test_swinunetr.csv,../ensemble/output_segmentations/hybrid_new/hybrid_patient_metrics_test.csv" --models "SegResNet,Attention UNet,SwinUNETR,Hybrid" --output "significant_results.csv"

# ensembles only
# python compare_models.py --csvs "../ensemble/output_segmentations/simple_avg/simple_avg_patient_metrics_test.csv,../ensemble/output_segmentations/perf_weight/perf_weight_patient_metrics_test.csv,../ensemble/output_segmentations/ttd/ttd_patient_metrics_test.csv,../ensemble/output_segmentations/hybrid_new/hybrid_new_patient_metrics_test.csv,../ensemble/output_segmentations/tta/tta_patient_metrics_test.csv" --models "Simple-Avg,Performance-Weighted,TTD,Hybrid,TTA" --output "significant_results_ensemble_models.csv"