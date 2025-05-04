from pathlib import Path

# base data directories
PROJECT_ROOT = Path(__file__).parents[1]
#GT_DIR = Path.home() / "data" / "brats2021challenge" / "RelabeledTrainingData"
GT_DIR = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\RelabeledTrainingData"
FIGURES_DIR = PROJECT_ROOT / "Figures"
UNC_MAPS_DIR = PROJECT_ROOT / "visualization" / "uncertainty_maps"
SIGNIFICANT_RESULTS = PROJECT_ROOT / "stats" / "results" / "significant_results_all_models.csv"
STATS_DIR = Path(__file__).resolve().parents[2] / "stats"

# Default CSVs
ENSEMBLE_OUT = PROJECT_ROOT / "ensemble" / "output_segmentations"
MODELS_OUT   = PROJECT_ROOT / "models"   / "performance"
ENSEMBLE_CSVS = [
    str(ENSEMBLE_OUT / "simple_avg/simple_avg_patient_metrics_test.csv"),
    str(ENSEMBLE_OUT / "perf_weight/perf_weight_patient_metrics_test.csv"),
    str(ENSEMBLE_OUT / "ttd/ttd_patient_metrics_test.csv"),
    str(ENSEMBLE_OUT / "hybrid_new/hybrid_new_patient_metrics_test.csv"),
    str(ENSEMBLE_OUT / "tta/tta_patient_metrics_test.csv"),
]

MODEL_CSVS = [
    # str(MODELS_OUT / "vnet/patient_metrics_test_vnet.csv"),
    str(MODELS_OUT / "segresnet/patient_metrics_test_segresnet.csv"),
    str(MODELS_OUT / "attunet/patient_metrics_test_attunet.csv"),
    str(MODELS_OUT / "swinunetr/patient_metrics_test_swinunetr.csv"),
]
DEFAULT_CSV_FILES = ENSEMBLE_CSVS + MODEL_CSVS

# Default model labels in the same order
DEFAULT_MODEL_NAMES = [
    "Simple-Avg",
    "Performance-Weighted",
    "TTD",
    "Hybrid",
    "TTA",
    "VNet",
    "SegResNet",
    "AttUNet",
    "SwinUNETR",
]

# plotting styles & fonts
STYLE = "seaborn-v0_8-whitegrid"
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 12

# color maps
METRIC_COLORS = {"HD95 NCR":"#1f77b4", "HD95 ED":"#ff7f0e", "HD95 ET":"#2ca02c"}
OVERLAY_COLORS = {1:(1,0,0,0.8), 2:(1,1,0,0.8), 3:(0,0,1,0.8)}

# Important‐cases definitions
CASE_SLICES = {
    "00332": 120,
    "01032": 115,
    "01147":  65,
    "01483":  80,
}
CASES = [
    ("00332","Unc Better"),
    ("01032","Simple Better"),
    ("01147","Good Case"),
    ("01483","Worst Case"),
]
INDIV_CASES = [
    ("01556","AttUNet Best"),
    ("01474","SegResNet Good"),
    ("01405","SwinUNETR Good"),
    ("01529","Borderline"),
]
SOURCES_ENSEMBLE = [
    ("simple_avg","SimpleAvg"),
    ("perf_weight","PerfWeight"),
    ("ttd","TTD"),
    ("tta","TTA"),
    ("hybrid_new","Hybrid"),
    ("gt","GT")
]
SOURCES_INDIV = [
    ("vnet","V-Net"),
    ("segresnet","SegResNet"),
    ("attunet","Attention UNet"),
    ("swinunetr","SwinUNETR"),
    ("gt","GT")
]


FEATURE_HEATMAP_CFG = {
    "metric": "ET",
    "model_display": {
        "simple_avg":  "Simple Avg.",
        "perf_weight": "Perf. Weight",
        "ttd":         "TTD",
        "tta":         "TTA",
        "hybrid_new":  "Hybrid",
    },
    "model_order": ["simple_avg","perf_weight","ttd","tta","hybrid_new"],
    "cmap": "coolwarm",
    "figsize": (24,12),
    "font_scale": 1.4,
}