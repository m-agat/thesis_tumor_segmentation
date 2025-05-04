import os
import os 
from viz_utils import * 

class ModelVisualizerBase:
    def __init__(self, model_type="ensemble"):
        self.model_type = model_type
        # root under which all patient folders live:
        self.outputs_root = "../predict/outputs"

        self.model_names = {
            "simple_avg":  "SimpleAvg",
            "perf_weight": "PerfWeight",
            "tta":         "TTA",
            "ttd":         "TTD",
            "hybrid":      "Hybrid",
            "segresnet":   "SegResNet",
            "attunet":     "AttUNet",
            "swinunetr":   "SwinUNETR",
            "gt":          "Ground Truth",
            "flair":       "FLAIR",
        }

        # which models to expect in each patient subfolder
        if model_type == "ensemble":
            self.models = ["simple_avg", "perf_weight", "tta", "ttd", "hybrid"]
        else:
            self.models = ["segresnet", "attunet", "swinunetr"]

    def get_model_path(self, model_name: str, patient_id: str) -> str:
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        # point at outputs/<patient_id>/<model_name>
        return os.path.join(self.outputs_root, patient_id, model_name)

    def load_prediction(self, model_name: str, patient_id: str):
        model_dir = self.get_model_path(model_name, patient_id)
        pred_file = os.path.join(model_dir, f"seg_{patient_id}.nii.gz")
        return load_nifti(pred_file)

    def load_ground_truth(self, patient_id):
        """Load ground truth for a patient."""
        gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"
        gt_seg_path = os.path.join(gt_base_path, f"{patient_id}_seg.nii.gz")
        return load_nifti(gt_seg_path)

    def load_flair(self, patient_id):
        """Load FLAIR image for a patient."""
        gt_base_path = f"/home/magata/data/braintumor_data/{patient_id}"
        flair_path = os.path.join(gt_base_path, "original", f"preprocessed1/preproc_{patient_id}_FLAIR_orig_skullstripped.nii.gz")
        return load_nifti(flair_path)

    def find_best_slice(self, patient_id, model_name=None, primary_label=1, fallback_labels=[2, 3]):
        """Find the best slice for visualization."""
        return find_slice_max_label(patient_id, model_name, self.get_model_path(model_name), primary_label, fallback_labels)