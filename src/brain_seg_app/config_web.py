import os

BASE_DIR = os.path.dirname(__file__)
RAW_UPLOADS = os.path.join(BASE_DIR, "assets", "raw", "uploads")
PREPROC_DIR  = os.path.join(BASE_DIR, "assets", "preprocessed")
SEG_DIR      = os.path.join(BASE_DIR, "assets", "segmentations")

# Streamlit settings
PAGE_TITLE = "🧠 Brain Tumor Segmentation"
LAYOUT     = "wide"

# Model / training settings
BATCH_SIZE = 1