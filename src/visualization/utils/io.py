from pathlib import Path
import pandas as pd
import nibabel as nib

def load_csvs(paths, names):
    """Load multiple CSVs, tag with a Model column, and concat."""
    dfs = []
    for p, n in zip(paths, names):
        df = pd.read_csv(p)
        df["Model"] = n
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_nifti(path: Path):
    """Load a NIfTI file and return its array."""
    return nib.load(str(path)).get_fdata()

def load_json(path: str) -> dict:
    """Read a JSON file and return its content as a dict."""
    import json
    with open(path, 'r') as f:
        return json.load(f)
    
def load_correlation_table(metric: str, ensemble: bool = True) -> pd.DataFrame:
    """
    Load the Spearman correlations CSV for the given metric.
    """
    from utils.constants import STATS_DIR
    fname = f"spearman_correlations_features_Dice_{metric}_{'ensemble' if ensemble else 'indiv'}.csv"
    path = STATS_DIR / "results" / fname
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df