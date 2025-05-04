import matplotlib.pyplot as plt
from typing import Dict, Sequence, Tuple
import pandas as pd
import numpy as np

def plot_ece_distributions(
    df: 'pd.DataFrame',
    regions: Sequence[str],
    ax: plt.Axes
):
    """
    Boxplot of ECE distributions per model & region using long-form DataFrame.
    Expects df with columns ['MODEL','REGION','ECE'].
    """
    models = df['MODEL'].unique()
    n_models = len(models)
    positions = np.arange(len(regions))
    width = 0.8 / n_models

    for i, model in enumerate(models):
        boxes = [
            df.loc[(df['MODEL'] == model) & (df['REGION'] == r), 'ECE'].dropna().values
            for r in regions
        ]
        pos = positions + (i - (n_models-1)/2) * width
        ax.boxplot(
            boxes,
            positions=pos,
            widths=width,
            patch_artist=True
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(regions)
    ax.set_xlabel('Subregion')
    ax.set_ylabel('ECE')
    ax.set_title('ECE Distribution per Model & Subregion')
    # legend via dummy artists
    for i, model in enumerate(models):
        ax.plot([], [], color='k', label=model)
    ax.legend(title='Model')

def plot_ece_summary(
    summary_df: 'pd.DataFrame',
    regions: Sequence[str],
    ax: plt.Axes
):
    """
    Bar plot with error bars of mean±std per model & region.
    Expects summary_df with ['MODEL','REGION','MEAN','STD'].
    """
    models = summary_df['MODEL'].unique()
    n_models = len(models)
    positions = np.arange(len(regions))
    width = 0.8 / n_models

    for i, model in enumerate(models):
        sub = summary_df[summary_df['MODEL'] == model]
        means = [sub.loc[sub['REGION'] == r, 'MEAN'].iloc[0] for r in regions]
        stds  = [sub.loc[sub['REGION'] == r, 'STD'].iloc[0] for r in regions]
        pos = positions + (i - (n_models-1)/2) * width
        ax.bar(pos, means, width, yerr=stds, capsize=5, label=model)
    ax.set_xticks(positions)
    ax.set_xticklabels(regions)
    ax.set_xlabel('Subregion')
    ax.set_ylabel('Mean ECE')
    ax.set_title('Average ECE ± STD per Model & Subregion')
    ax.legend(title='Model')

def scatter_error_vs_uncertainty(
    errors: np.ndarray,
    uncs: np.ndarray,
    region: str,
    ax: plt.Axes
):
    ax.scatter(uncs, errors, s=1, alpha=0.1)
    ax.set_yscale('log')
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Error (NLL)")
    ax.set_title(f"{region} (log-scale error)")

def binned_median_plot(
    errors: np.ndarray,
    uncs: np.ndarray,
    region: str,
    ax: plt.Axes,
    num_bins: int = 20
):
    edges = np.percentile(uncs, np.linspace(0,100,num_bins+1))
    centers, med_err = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (uncs >= lo) & (uncs < hi)
        centers.append((lo+hi)/2)
        med_err.append(np.median(errors[m]) if m.any() else np.nan)
    ax.plot(centers, med_err, marker='o')
    ax.set_xlabel("Uncertainty (bin center)")
    ax.set_ylabel("Median Error (NLL)")
    ax.set_title(f"Binned median Error for {region}")


def plot_histogram(
    values: np.ndarray,
    bins: int,
    xlabel: str,
    ylabel: str,
    title: str,
    ax: plt.Axes
):
    ax.hist(values, bins=bins, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_binned_error_curves(
    bin_centers: Dict[str, np.ndarray],
    avg_errors: Dict[str, np.ndarray],
    ax: plt.Axes,
    styles: Dict[str,Dict]
):
    """
    Plots one curve per region on the same Axes.
    `styles` maps region→{marker, color, label}.
    """
    for region, centers in bin_centers.items():
        ax.plot(
            centers,
            avg_errors[region],
            marker=styles[region]["marker"],
            linestyle="-",
            linewidth=2,
            markersize=5,
            label=f"{region} Avg Error",
        )
    ax.set_xlabel("Uncertainty (binned)")
    ax.set_ylabel("Avg Negative Log-Likelihood")
    ax.set_title("Uncertainty vs Error")
    ax.legend()
    ax.grid(True)

def plot_reliability_diagram(
    bin_stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    class_name: str,
    ax: plt.Axes
):
    centers, avg_conf, avg_acc, counts = bin_stats
    # accuracy curve
    ax.plot(centers, avg_acc, marker="o", linestyle="-", label="Observed")
    # perfect calibration line
    ax.plot([0,1],[0,1], "k--", label="Ideal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability: {class_name}")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc="upper left")
    ax.grid(True)
    # annotate counts
    for x,y,n in zip(centers, avg_acc, counts):
        if not np.isnan(y):
            ax.text(x, y, str(n), ha="center", va="bottom", fontsize=7)

def plot_risk_coverage_curve(
    ax: plt.Axes,
    coverage: Sequence[float],
    risk: Sequence[float],
    label: str,
    marker: str
):
    ax.plot(coverage, risk, marker=marker, linestyle='-',
            label=label)
    ax.set_xlabel("Coverage Fraction")
    ax.set_ylabel("Average Error (NLL)")
    ax.set_title("Risk–Coverage Curves")
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    ax.legend(title="Region")