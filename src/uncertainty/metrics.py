import numpy as np
import pandas as pd 
from scipy.stats import spearmanr
from uncertainty.constants import EPS
from typing import Tuple, List, Dict

def summarize_ece(
    df: pd.DataFrame,
    regions: List[str]
) -> pd.DataFrame:
    """
    Compute MEAN and STD of the ECE column for each MODEL × REGION.
    Returns columns ['MODEL','REGION','MEAN','STD'].
    """
    records = []
    # Ensure we have uppercase column names
    df = df.rename(columns=lambda c: c.upper())

    for model in df["MODEL"].unique():
        for region in regions:
            mask = (df["MODEL"] == model) & (df["REGION"] == region)
            sub = df.loc[mask, "ECE"]
            records.append({
                "MODEL": model,
                "REGION": region,
                "MEAN":  sub.mean(),
                "STD":   sub.std(),
            })

    return pd.DataFrame.from_records(records)

def voxel_error_and_uncertainty(
    prob: np.ndarray,
    gt: np.ndarray,
    unc_map: np.ndarray,
    region_label: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two 1-D arrays for voxels where gt == region_label:
      - error = -log p(label)
      - uncertainty from unc_map
    """
    mask = (gt == region_label)
    if not mask.any():
        return np.array([]), np.array([])
    p = prob[region_label, ...][mask]
    err = -np.log(p + EPS)
    uni = unc_map[mask]
    return err, uni

def spearman_error_uncertainty(
    errors: np.ndarray,
    uncs: np.ndarray
) -> Tuple[float, float]:
    """Compute Spearman’s rho and p-value (returns nan, nan if empty)."""
    if errors.size == 0:
        return np.nan, np.nan
    rho, p = spearmanr(errors, uncs)
    return float(rho), float(p)

def extract_region_error_uncertainty(
    prob: np.ndarray,
    gt: np.ndarray,
    unc_map: np.ndarray,
    region_label: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (errors, uncertainties) flattened for voxels where gt==region_label.
    Error = -log P(true_class) from `prob` (C,H,W,D).
    """
    mask = gt == region_label
    if not mask.any():
        return np.array([]), np.array([])
    p_true = prob[region_label][mask]
    return -np.log(p_true + EPS), unc_map[mask]

def bin_average_error(
    errors: np.ndarray,
    uncs: np.ndarray,
    num_bins: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bins `uncs` into equal-width bins and returns (bin_centers, avg_error).
    """
    if uncs.size == 0:
        return np.array([]), np.array([])
    edges = np.linspace(uncs.min(), uncs.max(), num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    avg_err: List[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (uncs >= lo) & (uncs < hi)
        avg_err.append(np.mean(errors[m]) if m.any() else np.nan)
    return centers, np.array(avg_err)

def compute_reliability_stats(
    confidences: np.ndarray,
    correctness: np.ndarray,
    num_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centers, avg_conf, avg_acc, counts) 
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    avg_conf = np.full(num_bins, np.nan, dtype=float)
    avg_acc  = np.full(num_bins, np.nan, dtype=float)
    counts   = np.zeros(num_bins, dtype=int)

    bin_idx = np.digitize(confidences, edges[:-1], right=False) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    for b in range(num_bins):
        mask = bin_idx == b
        cnt = mask.sum()
        counts[b] = cnt
        if cnt:
            avg_conf[b] = confidences[mask].mean()
            avg_acc[b]  = correctness[mask].mean()

    return centers, avg_conf, avg_acc, counts

def compute_risk_coverage(
    uncertainty: np.ndarray,
    error: np.ndarray,
    coverage_fractions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given flat arrays `uncertainty` and `error`, both length N,
    returns:
      - coverage_fractions (as passed in)
      - risk: average error over the fraction of voxels with lowest uncertainty
    """
    # sort by uncertainty ascending
    idx = np.argsort(uncertainty)
    u = uncertainty[idx]
    e = error[idx]
    n = len(u)
    risk = []
    for frac in coverage_fractions:
        k = int(n * frac)
        risk.append(np.mean(e[:k]) if k>0 else np.nan)
    return coverage_fractions, np.array(risk)