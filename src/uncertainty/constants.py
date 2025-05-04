from pathlib import Path
import numpy as np 

# === Directories ===
# Base data directory (override with CLI flags as needed)
DATA_DIR = Path("data")
# Subdirectories under DATA_DIR
PREDICTIONS_DIR = DATA_DIR / "predictions"
GROUND_TRUTH_DIR = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
# Output directories
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("Figures")

# === File name patterns ===
PREDICTION_PATTERN = "softmax_*.nii.gz"
GROUND_TRUTH_PATTERN = "{id}/{id}_seg.nii.gz"
UNCERTAINTY_PATTERN = "uncertainty_*.nii.gz"

# === Subregion labels ===
# Map human‐readable names to integer labels in segmentation maps
SUBREGIONS = {
    "NCR": 1,
    "ED": 2,
    "ET": 3,
}

# === Metric settings ===
# Number of bins for calibration/ECE calculations
ECE_BINS = 10
# Number of bins for error/uncertainty histograms
ERROR_BINS = 20
RELIABILITY_BINS = 10
COVERAGE_FRACTIONS = np.linspace(0.05, 1.0, 20)
# === Logging configuration ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# === Random seed for reproducibility ===
SEED = 42
EPS = 1e-8