import itertools
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def check_normality(df, group_col: str, metric: str) -> dict[str, float]:
    """
    Shapiro–Wilk per group. Returns {group: p_value}.
    """
    out = {}
    for grp, sub in df.groupby(group_col):
        vals = sub[metric].dropna()
        if len(vals) < 3:  # Shapiro needs ≥3
            out[grp] = np.nan
            continue
        _, p = stats.shapiro(vals)
        out[grp] = p
        print(f"{grp:12s} → p={p:.4f}")
    return out

def choose_test(df, group_col: str, metric: str, normality: dict[str, float]):
    """
    If *all* p>0.05 → ANOVA, else Kruskal. Returns (name, stat, p).
    """
    samples = [g[metric].dropna() for _, g in df.groupby(group_col)]
    if all((p or 0) > 0.05 for p in normality.values()):
        stat, p = stats.f_oneway(*samples)
        name = "ANOVA"
    else:
        stat, p = stats.kruskal(*samples)
        name = "Kruskal"
    print(f"{name}: stat={stat:.4f}, p={p:.4f}")
    return name, stat, p

def posthoc_anova(df, group_col: str, metric: str, alpha=0.05):
    """
    Tukey’s HSD. Returns list of dicts for significant pairs.
    """
    tuk = pairwise_tukeyhsd(df[metric].dropna(), df[group_col].dropna(), alpha)
    sig = []
    for g1, g2, meandiff, p, _, _ in tuk.summary().data[1:]:
        if p < alpha:
            sig.append((g1, g2, meandiff, p))
    return sig

def posthoc_nonparametric(df, group_col: str, metric: str, alpha=0.05):
    """
    Pairwise Mann–Whitney U with Bonferroni. 
    """
    grps = list(df[group_col].unique())
    pairs = itertools.combinations(grps, 2)
    sig = []
    m = 0
    for g1, g2 in pairs:
        a = df.loc[df[group_col]==g1, metric].dropna()
        b = df.loc[df[group_col]==g2, metric].dropna()
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Bonferroni
        adj_p = min(p * (len(grps)*(len(grps)-1)/2), 1.0)
        if adj_p < alpha:
            sig.append((g1, g2, stat, adj_p))
        m += 1
    return sig
