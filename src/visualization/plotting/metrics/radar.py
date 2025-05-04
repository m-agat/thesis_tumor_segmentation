import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_radar(
    stats: dict[str,dict],
    model_names: list[str],
    metrics: list[str],
    colors: list[str],
    out_path: str
):
    """
    Radar chart for multiple metrics per model.
    - stats: { model: {metric: value} }
    """
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for i, model in enumerate(model_names):
        vals = [stats[model][m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, label=model, color=colors[i])
        ax.fill(angles, vals, alpha=0.25, color=colors[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
