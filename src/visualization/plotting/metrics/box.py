import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_box(
    base_path: str,
    group_names: dict[str,str],
    metrics: list[str],
    plot_type: str,
    file_pattern: str,
    out_combined: str,
    out_individual: str
):
    """
    Generic boxplot for patient-level metrics.
    - file_pattern: e.g. "{key}_patient_metrics_test.csv"
    """
    # collect data
    data = {m: {'data': [], 'labels': []} for m in metrics}
    for key,label in group_names.items():
        csv = os.path.join(base_path, key, file_pattern.format(key=key))
        df = pd.read_csv(csv)
        for m in metrics:
            data[m]['data'].append(df[m])
            data[m]['labels'].append(label)

    # combined
    fig, axs = plt.subplots(1, len(metrics), figsize=(18,6), sharey=True)
    for i,m in enumerate(metrics):
        axs[i].boxplot(data[m]['data'], labels=data[m]['labels'])
        axs[i].set_title(m)
        if i>0: axs[i].set_ylabel('')
    plt.tight_layout()
    plt.savefig(out_combined, dpi=300)
    plt.close(fig)

    # individual
    for m in metrics:
        fig,ax = plt.subplots(figsize=(8,6))
        ax.boxplot(data[m]['data'], labels=data[m]['labels'])
        ax.set_title(m)
        plt.tight_layout()
        fname = m.replace(' ','_').lower()
        plt.savefig(out_individual.format(fname), dpi=300)
        plt.close(fig)
