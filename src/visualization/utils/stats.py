import pandas as pd

def load_metrics(csv_files: list[str], model_names: list[str]) -> pd.DataFrame:
    """Load CSVs, tag with Model, concat."""
    dfs = []
    for f, m in zip(csv_files, model_names):
        df = pd.read_csv(f)
        df["Model"] = m
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def summarize(df: pd.DataFrame, metrics: list[str]) -> dict[str, dict]:
    """Compute mean & SEM per ModelxMetric."""
    out = {}
    grp = df.groupby("Model")
    for metric in metrics:
        s = grp[metric]
        out[metric] = {
            "means": s.mean(),
            "sems":  s.sem()
        }
    return out

def melt_for_seaborn(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Wide→long for seaborn catplot."""
    return df.melt(
        id_vars="Model",
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )

def load_significant_pairs(path: str) -> dict[str, list[tuple[str,str]]]:
    df = pd.read_csv(path)
    out = {}
    for metric, g in df.groupby("metric"):
        out[metric] = list(zip(g["group1"], g["group2"]))
    return out


def compute_group_stats(df: pd.DataFrame, metrics: list[str]) -> tuple[pd.Series, pd.Series]:
    """
    Given a DataFrame with columns ['Model'] + metrics, compute mean and sem for each metric per Model.
    Returns two Series: means (indexed by Model) and sems (indexed by Model).
    """
    means = df.groupby('Model')[metrics].mean()
    sems  = df.groupby('Model')[metrics].sem()
    return means, sems