import brain_seg_app.model as fem
import torch
from streamlit import cache_resource

@cache_resource(show_spinner=False)
def load_models():
    return fem.load_all_models()

def run_ensemble(test_loader, models, output_dir, progress_bar=None):
    """
    Wrapper around fem.ensemble_segmentation that accepts a Streamlit
    progress widget to show model‐level progress.
    """
    return fem.ensemble_segmentation(
        test_loader,
        models,
        composite_score_weights={"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1},
        n_iterations=10,
        progress_bar=progress_bar,
        output_dir=output_dir
    )
