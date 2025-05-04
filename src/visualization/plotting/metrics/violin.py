import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_violin(
    base_path: str,
    group_names: dict[str,str],
    metrics: list[str],
    file_pattern: str,
    out_combined: str,
    out_individual: str
):
    """
    Generic violin plot for patient-level metrics.
    """
    # collect values
    data = {m: {'values': [], 'labels': []} for m in metrics}
    for key,label in group_names.items():
        csv = os.path.join(base_path, key, file_pattern.format(key=key))
        df = pd.read_csv(csv)
        for m in metrics:
            data[m]['values'].extend(df[m])
            data[m]['labels'].extend([label]*len(df))

    # combined
    fig,axs = plt.subplots(1,len(metrics),figsize=(18,6),sharey=True)
    for i,m in enumerate(metrics):
        sns.violinplot(x=data[m]['labels'], y=data[m]['values'], ax=axs[i], inner='box', cut=0)
        if i>0: axs[i].set_ylabel('')
    plt.tight_layout()
    plt.savefig(out_combined, dpi=300)
    plt.close(fig)

    # individual
    for m in metrics:
        fig,ax = plt.subplots(figsize=(8,6))
        sns.violinplot(x=data[m]['labels'], y=data[m]['values'], ax=ax, inner='box', cut=0)
        ax.set_title(m)
        plt.tight_layout()
        fname=m.replace(' ','_').lower()
        plt.savefig(out_individual.format(fname), dpi=300)
        plt.close(fig)