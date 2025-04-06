import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# --------------------------
# 1. Load your data
# --------------------------
patient_uncertainties_path = "./outputs/uncertainties/patient_uncertainties_minmax_val.csv"
patient_uncertainties = pd.read_csv(patient_uncertainties_path)

performance_data_path = "../models/performance/segresnet/composite_scores.csv"
performance_data = pd.read_csv(performance_data_path)

# --------------------------
# 2. Select model/subregion/uncertainty
# --------------------------
model = "segresnet"
uncertainty_method = "TTA"
subregion = "ED"

# Extract columns
uncertainty_col = f"{uncertainty_method} {subregion}"
segresnet_uncertainties = patient_uncertainties.loc[
    patient_uncertainties["Model"] == model, uncertainty_col
].values
scores = performance_data[subregion].values

# --------------------------
# 3. Define an outlier detection function
# --------------------------
def identify_outliers_iqr(values):
    """Return a boolean mask where True indicates an outlier."""
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (values < lower_bound) | (values > upper_bound)

# Identify outliers on x (uncertainty) and y (scores)
x_outliers = identify_outliers_iqr(segresnet_uncertainties)
y_outliers = identify_outliers_iqr(scores)
combined_outliers = x_outliers | y_outliers  # mark if outlier in either dimension

# Separate inliers/outliers
x_inliers = segresnet_uncertainties[~combined_outliers]
y_inliers = scores[~combined_outliers]
x_outs = segresnet_uncertainties[combined_outliers]
y_outs = scores[combined_outliers]

# --------------------------
# 4. Plot scatter + outliers
# --------------------------
plt.figure(figsize=(8, 6))

# Plot inliers in blue, outliers in red
plt.scatter(x_inliers, y_inliers, color='blue', alpha=0.7, label='Inliers')
plt.scatter(x_outs,   y_outs,   color='red',  alpha=0.7, label='Outliers')

# Fit line on ALL data
slope_all, intercept_all = np.polyfit(segresnet_uncertainties, scores, 1)
x_range = np.linspace(segresnet_uncertainties.min(), segresnet_uncertainties.max(), 100)
y_fit_all = slope_all * x_range + intercept_all
plt.plot(x_range, y_fit_all, color="green", linestyle='--', label='Fit (All Data)')

# Fit line on INLIERS only
if len(x_inliers) > 1:  # Ensure we have enough points to fit
    slope_in, intercept_in = np.polyfit(x_inliers, y_inliers, 1)
    y_fit_in = slope_in * x_range + intercept_in
    plt.plot(x_range, y_fit_in, color="black", linestyle='-', label='Fit (Inliers)')

# --------------------------
# 5. Calculate Spearman Correlations
# --------------------------
spearman_all, pval_all = spearmanr(segresnet_uncertainties, scores)
spearman_in, pval_in = spearmanr(x_inliers, y_inliers)

# --------------------------
# 6. Finalize plot
# --------------------------
plt.xlabel(f"{uncertainty_method} {subregion} Uncertainty")
plt.ylabel("Composite Score")
plt.title(f"{model}: Composite Score vs {uncertainty_method} {subregion} Uncertainty")
plt.grid(True)
plt.legend()

# Annotate correlation values
text_str = (
    f"Spearman (All): {spearman_all:.3f} (p={pval_all:.3e})\n"
    f"Spearman (Inliers): {spearman_in:.3f} (p={pval_in:.3e})"
)
plt.text(
    0.05, 0.05, text_str,
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7)
)

plt.show()
