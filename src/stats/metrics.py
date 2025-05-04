import pandas as pd

def add_region_averages(
    dfs: dict[str, pd.DataFrame],
    metrics: list[str],
    regions: list[str],
    suffix: str = "avg_regions"
) -> None:
    """
    In-place, for each df in dfs, compute
      df[f"{metric} {suffix}"] = mean over [f"{metric} {r}" for r in regions]
    """
    for df in dfs.values():
        for metric in metrics:
            cols = [f"{metric} {r}" for r in regions]
            df[f"{metric} {suffix}"] = df[cols].mean(axis=1)

def summarize_metrics(
    dfs: dict[str, pd.DataFrame],
    metrics: list[str],
    regions: list[str],
    avg_suffix: str = "avg_regions"
) -> pd.DataFrame:
    """
    Build a long-form summary table:
       model | metric         | region         | mean | std
       ------|----------------|----------------|------|----
       V-Net | Dice           | NCR            | 0.82 | 0.05
       V-Net | Dice           | avg_regions    | 0.79 | 0.04
       ...
    """
    rows = []
    for model, df in dfs.items():
        for metric in metrics:
            for region in regions + [avg_suffix]:
                col = (f"{metric} {region}" if region != avg_suffix
                       else f"{metric} {avg_suffix}")
                vals = df[col].dropna()
                rows.append({
                    "model":    model,
                    "metric":   metric,
                    "region":   region,
                    "mean":     vals.mean(),
                    "std":      vals.std()
                })
    return pd.DataFrame(rows)
