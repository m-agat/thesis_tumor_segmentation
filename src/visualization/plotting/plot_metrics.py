import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.utils.stats import load_metrics, summarize, melt_for_seaborn
from visualization.utils.plotting_helpers import add_significance_markers

def plot_metrics(
    csv_files: list[str],
    model_names: list[str],
    metrics: list[str],
    palette: dict[str,str] | None = None,
    sig_csv: str | None = None,
    figsize=(20, 8),
    out_path: str | None = None
):
    """
    Generic bar‐grid of any metrics, rendered exactly like your old HD95 script.
    - metrics: list of column names, e.g. ["HD95 NCR","Dice","Sensitivity"]
    - palette: { metric_name: color } or None for default HD95‐style colors
    - sig_csv: CSV with cols [metric, group1, group2] for markers
    - out_path: if provided, where to save the figure (PNG)
    """
    # 1) load & summarize
    df      = load_metrics(csv_files, model_names)
    stats   = summarize(df, metrics)
    long_df = melt_for_seaborn(df, metrics)

    # 2) style + default colors (HD95 palette if none)
    plt.style.use("seaborn-v0_8-whitegrid")
    if palette is None:
        # replicate your old blue/orange/green for the first 3 metrics
        default = ['#1f77b4', '#ff7f0e', '#2ca02c']
        if len(metrics) == 3:
            palette = dict(zip(metrics, default))
        else:
            palette = dict(zip(metrics, sns.color_palette("tab10", len(metrics))))

    # 3) build subplots
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # means & sems in the right order
        means = stats[metric]["means"].reindex(model_names)
        sems  = stats[metric]["sems"].reindex(model_names)

        x = np.arange(len(model_names))
        # 4) bar chart with caps
        bars = ax.bar(
            x, means, yerr=sems, capsize=5,
            color=palette[metric]
        )

        # 5) numeric labels just above each bar
        for bar, m, s in zip(bars, means, sems):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + s + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{m:.3f}",
                ha="center", va="bottom", fontsize=12
            )

        # 6) formatting
        ax.set_title(metric, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(
            model_names, rotation=45, ha="right", fontsize=12
        )
        if ax is axes[0]:
            # only leftmost has the y‐label
            base = metric.split()[0]
            ax.set_ylabel(f"{base} Score", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)

        # 7) add significance bars if requested
        if sig_csv:
            from ..utils.stats import load_significant_pairs
            sigs = load_significant_pairs(sig_csv).get(metric, [])
            add_significance_markers(ax, means, sems, sigs, model_names)

    fig.tight_layout()

    # 8) save & show
    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {out_path}")
    plt.show()

    return fig, axes
