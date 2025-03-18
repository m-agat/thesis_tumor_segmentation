import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import time
import pandas as pd
import json
import re
import math 

# Add custom modules
sys.path.append("../")
import models.models as models
import config.config as config
from utils.utils import AverageMeter
from uncertainty.test_time_augmentation import tta_variance
from uncertainty.test_time_dropout import ttd_variance, minmax_uncertainties
from dataset import dataloaders

#####################
#### Load Models ####
#####################


def load_model(model_class, checkpoint_path, device):
    """
    Load a segmentation model from a checkpoint.
    """
    model = model_class
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, partial(
        sliding_window_inference,
        roi_size=config.roi,
        sw_batch_size=config.sw_batch_size,
        predictor=model,
        overlap=config.infer_overlap,
    )


def load_all_models():
    """
    Load all segmentation models into a dictionary.
    """
    return {
        "swinunetr": load_model(
            models.swinunetr_model, config.model_paths["swinunetr"], config.device
        ),
        "segresnet": load_model(
            models.segresnet_model, config.model_paths["segresnet"], config.device
        ),
        "attunet": load_model(
            models.attunet_model, config.model_paths["attunet"], config.device
        ),
        "vnet": load_model(
            models.vnet_model, config.model_paths["vnet"], config.device
        ),
    }

#####################
#### Load Weights ####
#####################

def load_weights(performance_weights_path):
    with open(performance_weights_path) as f:
        performance = json.load(f)
    return performance 


def extract_patient_id(path):
    # Use regular expression to find all numbers in the path
    numbers = re.findall("\d+", path)

    # Assuming the patient ID is the last number found
    patient_id = numbers[-1]

    return patient_id

def save_metrics_csv(metrics_list, filename):
    """
    Save per-patient segmentation performance metrics to CSV.
    """
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)

    print(f"Saved patient-wise metrics to {filename}")


def save_average_metrics(metrics_list, filename):
    """
    Save the average test set performance in a JSON file.
    """
    avg_metrics = {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in metrics_list[0]
        if key != "patient_id"
    }

    with open(filename, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print(f"Saved average test set metrics to {filename}")


def summarize_uncertainty(
    data_loader, models_dict, n_iterations=20, patient_id=None, output_dir="./outputs/uncertainties"
):
    """
    Perform segmentation using an ensemble of multiple models with simple averaging.

    Parameters:
    - patient_id: ID of the patient whose scan is being segmented.
    - data_loader: Dataloader for the test set.
    - models_dict: Dictionary containing trained models and their inferers.
    - output_dir: Directory where segmentations will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    if patient_id is not None:
        # Get a subset of test loader with the specific patient
        data_loader = config.find_patient_by_id(patient_id, data_loader)
    else:
        # Get full test data loader
        data_loader = data_loader
    
    uncertainty_data = []
    with torch.no_grad():
        for batch_data in data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            patient_id = extract_patient_id(reference_image_path)
            gt = batch_data["label"].to(
                config.device
            )  # shape: (batch_size, 240, 240, 155)

            print(
                f"\nProcessing patient: {patient_id}\n",
            )
            tta_uncertainties = {patient_id: {}}
            ttd_uncertainties = {patient_id: {}}
            hybrid_uncertainties = {patient_id: {}}

            w_ttd = 0.5
            w_tta = 0.5
            for model_name, (model, inferer) in models_dict.items():
                # Get dropout uncertainty (TTD)
                ttd_mean, ttd_uncertainty = ttd_variance(model, inferer, image, config.device, n_iterations=n_iterations)
                # Get augmentation uncertainty (TTA)
                tta_mean, tta_uncertainty = tta_variance(inferer, image, config.device, n_iterations=n_iterations)
                
                # Normalize each uncertainty map (assumes minmax_uncertainties scales to 0-1).
                ttd_uncertainty_norm = minmax_uncertainties(ttd_uncertainty)
                tta_uncertainty_norm = minmax_uncertainties(tta_uncertainty)
                
                # Compute the hybrid uncertainty by averaging the normalized uncertainties.
                hybrid_uncertainty = w_ttd * ttd_uncertainty_norm + w_tta * tta_uncertainty_norm
                
                # Remove the batch dimension: now shape becomes (num_classes, H, W, D)
                tta_uncertainty_norm = np.power(np.squeeze(tta_uncertainty_norm, axis=0), 0.5)
                ttd_uncertainty_norm = np.power(np.squeeze(ttd_uncertainty_norm, axis=0), 0.5)
                hybrid_uncertainty = np.power(np.squeeze(hybrid_uncertainty, axis=0), 0.5)

                uncertainty_data.append({
                    "Patient ID": patient_id,
                    "Model": model_name,
                    "TTA NCR": np.median(tta_uncertainty_norm[1]),
                    "TTA ED": np.median(tta_uncertainty_norm[2]),
                    "TTA ET": np.median(tta_uncertainty_norm[3]),
                    "TTD NCR": np.median(ttd_uncertainty_norm[1]),
                    "TTD ED": np.median(ttd_uncertainty_norm[2]),
                    "TTD ET": np.median(ttd_uncertainty_norm[3]),
                    "Hybrid NCR": np.median(hybrid_uncertainty[1]),
                    "Hybrid ED": np.median(hybrid_uncertainty[2]),
                    "Hybrid ET": np.median(hybrid_uncertainty[3]),
                })
                
            torch.cuda.empty_cache()

    df = pd.DataFrame(uncertainty_data)
    csv_path = os.path.join(output_dir, "patient_uncertainties.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved per-patient uncertainty medians to: {csv_path}")

####################################
#### Visualize Segmentation ####
####################################


def visualize_segmentation(segmentation, patient_id):
    """
    Display and save a middle slice from the segmentation.

    Parameters:
    - segmentation: 3D NumPy array containing the segmentation result.
    - patient_id: ID of the patient (used for file naming).
    """
    slice_index = segmentation.shape[-1] // 2  # Middle slice

    plt.figure(figsize=(6, 6))
    plt.imshow(segmentation[:, :, slice_index], cmap="gray")
    plt.title(f"Segmentation Slice at Index {slice_index}")
    plt.axis("off")
    plt.savefig(f"tta_ttd_segmentation_{patient_id}_slice.png")
    # plt.show()


#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "01331"
    _, val_loader = dataloaders.get_loaders(
        batch_size=config.batch_size,
        json_path=config.json_path,
        basedir=config.root_dir,
        fold=None,
        roi=config.roi,
        use_final_split=True,
    )
    models_dict = load_all_models()
    summarize_uncertainty(val_loader, models_dict, n_iterations=20, patient_id=patient_id)
