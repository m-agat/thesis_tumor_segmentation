import pandas as pd
from scipy.stats import spearmanr
import argparse
import pandas as pd
import os 
from stats.data import load_feature_table, build_model_paths, load_model_performances
from stats.utils import ensure_dir

def compute_spearman_correlations(
    perf_df: pd.DataFrame,
    features_df: pd.DataFrame,
    metric: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Merge on patient_id, then for each column in features_df (except 'patient_id'),
    compute Spearman’s rho vs. perf_df[metric]. Returns a DataFrame with:
      ['Feature','Spearman','p_value'] filtered to p_value < alpha.
    """
    merged = pd.merge(perf_df, features_df, on="patient_id", how="inner")
    rows = []
    scores = merged[metric]
    for feat in features_df.columns:
        if feat == "patient_id":
            continue
        vals = merged[feat]
        rho, p = spearmanr(vals, scores, nan_policy="omit")
        rows.append({
            "Feature": feat,
            "Spearman": rho,
            "p_value": p
        })
    df_corr = pd.DataFrame(rows)
    return df_corr[df_corr["p_value"] < alpha].reset_index(drop=True)

def main(
    feature_csv: str,
    models: str,
    perf_base: str,
    metric: str,
    alpha: float,
    out_csv: str
):
    # 1) Load
    features = load_feature_table(feature_csv)
    # build a { model_name: csv_path } map
    model_list = [m.strip() for m in models.split(",")]
    path_map = build_model_paths(
                   model_list,
                   perf_base,
                   metrics_filename="{model}_patient_metrics_test.csv"
               )
    perfs = load_model_performances(path_map)

    # 2) Compute & collect
    all_results = []
    for model, df in perfs.items():
        print(f"\n→ Computing correlations for model: {model}")
        corr_df = compute_spearman_correlations(df, features, metric, alpha)
        corr_df.insert(0, "Model", model)
        all_results.append(corr_df)

    # 3) Combine & save
    if all_results:
        result = pd.concat(all_results, ignore_index=True)
        ensure_dir(os.path.dirname(out_csv) or ".")
        result.to_csv(out_csv, index=False)
        print(f"\nSaved correlations to {out_csv}")
    else:
        print("No significant correlations found or no data loaded.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute Spearman correlations between image features and a performance metric"
    )
    p.add_argument("--features",   required=True,
                   help="CSV of extracted features (must include patient_id)")
    p.add_argument("--models",     required=True,
                   help="Comma-separated list of model folder names")
    p.add_argument("--perf-base",  default="../ensemble/output_segmentations",
                   help="Base path where model folders live")
    p.add_argument("--metric",     default="Dice ET",
                   help="Performance column to correlate (e.g. 'Dice ET')")
    p.add_argument("--alpha",      type=float, default=0.05,
                   help="Significance threshold for p-values")
    p.add_argument("--output",     required=True,
                   help="Path to save the CSV of significant correlations")
    args = p.parse_args()

    main(
        feature_csv=args.features,
        models=args.models,
        perf_base=args.perf_base,
        metric=args.metric,
        alpha=args.alpha,
        out_csv=args.output
    )