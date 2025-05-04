import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from visualization.utils.io import load_json

sns.set_style("whitegrid")

def plot_grouped_bar(
    base_path: str,
    group_names: dict[str,str],
    metrics: list[str],
    region_colors: dict[str,str],
    json_file: str,
    csv_file: str,
    out_path: str
):
    """
    Generic grouped bar plot for average+std metrics.
    - base_path: root folder containing model subfolders
    - group_names: { subfolder: display_name }
    - metrics: list of metric column names
    - region_colors: { region: color }
    - json_file: name of JSON with average metrics
    - csv_file: name of CSV with per-patient metrics
    - out_path: file to save the plot
    """
    # load averages
    model_avgs = {}
    for key in group_names:
        data = load_json(os.path.join(base_path, key, json_file))
        model_avgs[key] = [data[m] for m in metrics]

    # load stds
    model_stds = {m: [] for m in metrics}
    for key in group_names:
        csv_path = os.path.join(base_path, key, csv_file.format(key=key))
        df = pd.read_csv(csv_path)
        for m in metrics:
            model_stds[m].append(df[m].std())

    # prepare bar positions
    x = np.arange(len(group_names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(14,8))
    for i, m in enumerate(metrics):
        vals = [model_avgs[k][i] for k in group_names]
        errs = model_stds[m]
        ax.bar(
            x + i*width, vals, width,
            yerr=errs, capsize=5,
            label=m.split()[-1], color=region_colors.get(m.split()[-1])
        )

    ax.set_xticks(x + width*(len(metrics)-1)/2)
    ax.set_xticklabels(group_names.values(), rotation=45)
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.legend(title='Sub-region')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

