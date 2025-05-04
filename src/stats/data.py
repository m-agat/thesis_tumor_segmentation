import os 
import pandas as pd
import os
import pandas as pd
from typing import Callable, Dict, List, Optional, Any
from stats.utils import format_patient_id_csv, format_patient_id

def load_ground_truth(path: str) -> pd.DataFrame:
    """
    Load and normalize the ground-truth region presence CSV.
    Expects columns: Patient, NCR present, ED present, ET present, Region Combination.
    Returns a DataFrame with a uniform patient_id column.
    """
    gt = pd.read_csv(path)
    gt['patient_id'] = gt['Patient'].apply(format_patient_id)
    # Keep only the bits we care about
    return gt[['patient_id', 'NCR present', 'ED present', 'ET present', 'Region Combination']]

def merge_with_gt(model_df: pd.DataFrame, gt_df: pd.DataFrame,
                  on: str = 'patient_id', how: str = 'inner') -> pd.DataFrame:
    """Quick merge helper to join performance with ground truth."""
    return pd.merge(model_df, gt_df, on=on, how=how)

def load_data(csv_files: list[str], model_names: list[str]) -> pd.DataFrame:
    """
    Reads each CSV, tags it with its model name, and concatenates.
    Skips missing files with a warning.
    """
    dfs = []
    for path, name in zip(csv_files, model_names):
        if not os.path.isfile(path):
            # you might swap print→logging.warning
            print(f"[WARNING] Missing file: {path}")
            continue
        df = pd.read_csv(path)
        df["Model"] = name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def get_numeric_metrics(df: pd.DataFrame, exclude: list[str] = None) -> list[str]:
    """
    Returns all float/int columns except those in exclude.
    By default excludes ['Model', 'Patient'].
    """
    exclude = set(exclude or ["Model", "Patient"])
    numeric = df.select_dtypes(include=["int64", "float64"]).columns
    return [c for c in numeric if c not in exclude]

def load_feature_table(path: str) -> pd.DataFrame:
    """
    Reads the features CSV and ensures 'patient_id' exists.
    """
    df = pd.read_csv(path, dtype={"patient_id": str})
    if 'patient_id' not in df.columns:
        raise KeyError("Expected 'patient_id' column in features CSV")
    df['patient_id'] = df['patient_id'].str.extract(r'(\d+)$')[0].str.zfill(5)
    return df

def build_model_paths(
    model_names: List[str],
    base_dir: str,
    metrics_filename: str = "{model}_patient_metrics_test.csv",
) -> Dict[str, str]:
    """
    Given a list of model names and a base directory, return a dict
    mapping each model to its metrics CSV path.
    """
    return {
        model: os.path.join(base_dir, model, metrics_filename.format(model=model))
        for model in model_names
    }

def load_model_performances(
    path_map: Dict[str, str],
    id_col: str = "patient_id",
    id_formatter: Callable[[Any], str] = format_patient_id_csv,
    how: str = "inner",
) -> Dict[str, pd.DataFrame]:
    """
    Load one or more model-performance CSVs into DataFrames.

    Args:
      path_map: dict of { model_name: csv_path }
      id_col:  name of the column to re-format (default "patient_id")
      id_formatter:  function to normalize that column
      how:  how to handle missing files ("warn" vs. "error" vs. "ignore")

    Returns:
      dict of { model_name: DataFrame } (only those that could be loaded)
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for model, path in path_map.items():
        if not os.path.isfile(path):
            msg = f"[WARN] Missing performance file for {model!r}: {path}"
            if how == "error":
                raise FileNotFoundError(msg)
            elif how == "warn":
                print(msg)
                continue
            else:  # ignore
                continue
        df = pd.read_csv(path)
        df[id_col] = df[id_col].apply(id_formatter)
        dfs[model] = df
    return dfs