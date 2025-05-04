import seaborn as sns
import matplotlib.pyplot as plt
from utils.constants import FIGURES_DIR, FEATURE_HEATMAP_CFG
from utils.io import load_correlation_table
import argparse
from pathlib import Path 

def plot_feature_performance_heatmap(
    metric: str,
    model_display: dict[str,str],
    model_order: list[str],
    cmap: str,
    figsize: tuple[float,float],
    font_scale: float,
    out_path: Path | None = None
):
    # 1) load table
    df = load_correlation_table(metric)

    # 2) pivot to heatmap form
    heat = (
        df
        .pivot_table(values="Spearman Correlation",
                     index="Feature", columns="Model", aggfunc="mean")
    )

    # 3) reorder & rename columns
    avail = [m for m in model_order if m in heat.columns]
    heat = heat[avail].rename(columns=model_display)

    # 4) plot
    plt.figure(figsize=figsize)
    sns.set_context("notebook", font_scale=font_scale)
    ax = sns.heatmap(
        heat, annot=True, fmt=".2f", cmap=cmap, center=0,
        linewidths=0.5
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(f"Spearman Correlation: Dice {metric}", fontsize=26, pad=20)
    ax.set_xlabel("Model",   fontsize=24)
    ax.set_ylabel("Feature", fontsize=24)
    plt.tight_layout()

    out = out_path or (FIGURES_DIR / f"features_perf_heatmap_{metric}_ensemble.png")
    out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out



def main():
    p = argparse.ArgumentParser(description="Feature vs. performance heatmap")
    p.add_argument(
      "--metric", "-k",
      default=FEATURE_HEATMAP_CFG["metric"],
      help="Which sub-region metric (e.g. ET, ED, NCR)"
    )
    p.add_argument(
      "--out", "-o",
      help="Where to save the heatmap PNG"
    )
    args = p.parse_args()

    cfg = FEATURE_HEATMAP_CFG.copy()
    cfg["metric"] = args.metric
    out = plot_feature_performance_heatmap(
      metric        = cfg["metric"],
      model_display = cfg["model_display"],
      model_order   = cfg["model_order"],
      cmap          = cfg["cmap"],
      figsize       = cfg["figsize"],
      font_scale    = cfg["font_scale"],
      out_path      = args.out
    )
    print("Saved heatmap to", out)

if __name__=="__main__":
    main()
