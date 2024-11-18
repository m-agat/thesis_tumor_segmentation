import os
import sys
sys.path.append("../")
import torch
import gc
import config.config as config
import models.models as models
from monai.inferers import sliding_window_inference
from sklearn.metrics import roc_auc_score, roc_curve
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_metrics_with_monai,
    calculate_composite_score,
)
from monai.metrics import ConfusionMatrixMetric
import psutil 
from torch.cuda.amp import autocast
import tracemalloc
import pickle

def print_memory_usage():
    print(f"RAM Memory: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.2f}%")

print("Initial memory usage:")
print_memory_usage()

# Load model function
def load_model():
    print(f"Loading model: {config.model_name}")
    model_map = {
        "swinunetr": lambda: models.swinunetr_model,
        "segresnet": lambda: models.segresnet_model,
        "vnet": lambda: models.vnet_model,
        "attunet": lambda: models.attunet_model,
    }
    model = model_map[config.model_name]()
    checkpoint = torch.load(config.model_file_path, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(config.device).eval()
    print(f"Model {config.model_name} loaded and moved to {config.device}")
    return model

model = load_model()
print("Memory usage after loading model:")
print_memory_usage()
model_inferer_test = partial(
    sliding_window_inference,
    roi_size=config.roi,
    sw_batch_size=config.sw_batch_size,
    predictor=model,
    overlap=config.infer_overlap,
)

class_labels = [0, 1, 2, 4]
output_channel_to_class_label = {0: 1, 1: 2, 2: 4}
weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
patient_scores = []
total_patients = len(config.test_loader_patient)
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False
)

# Intermediate result directory
intermediate_dir = os.path.join(config.output_dir, "intermediate_tensors")
roc_data_dir = os.path.join(config.output_dir, "roc_data")
os.makedirs(roc_data_dir, exist_ok=True)

os.makedirs(intermediate_dir, exist_ok=True)
tracemalloc.start()

print("Starting inference and metric computation")

save_interval=2
results_save_path = os.path.join(config.output_dir, f"patient_performance_scores_1339_{config.model_name}.csv")

with torch.no_grad():
    for idx, batch_data in enumerate(config.test_loader_patient):
        image = batch_data["image"].to(config.device)
        patient_path = batch_data["path"]

        print(f"Processing patient {idx}/{total_patients}: {patient_path[0]}")
        print("Memory usage at start of patient processing:")
        print_memory_usage()
        with autocast():
            prob = torch.sigmoid(model_inferer_test(image))
        del image
        gc.collect()
        # print(f"Probability output shape for patient {patient_path[0]}: {prob.shape}")
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        # print(f"Segmentation output for patient {patient_path[0]}: {seg.shape[0]} channels found.")
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4

        ground_truth = batch_data["label"][0].cpu().numpy()

        patient_metrics = {"Patient": patient_path[0]}
        for tissue_type in class_labels:
            print(f"  Computing metrics for tissue type {tissue_type}")
            # Compute and store metrics
            try:
                dice_score = compute_dice_score_per_tissue(seg_out, ground_truth, tissue_type)
                print(f"    Dice score for tissue {tissue_type}: {dice_score}")
            except Exception as e:
                print(f"    Error computing Dice for tissue {tissue_type}: {e}")
                dice_score = np.nan
            try:
                hd95 = compute_hd95(seg_out, ground_truth, tissue_type)
                if np.isnan(hd95):
                    print(f"    HD95 for tissue {tissue_type} not computable, setting to 0")
                    hd95 = float('inf')
                print(f"    HD95 for tissue {tissue_type}: {hd95}")
            except Exception as e:
                print(f"    Error computing HD95 for tissue {tissue_type}: {e}")
                hd95 = np.nan
            try:
                sensitivity, specificity, f1_score = compute_metrics_with_monai(seg_out, ground_truth, tissue_type, confusion_metric)
                print(f"    Sensitivity for tissue {tissue_type}: {sensitivity}")
                print(f"    Specificity for tissue {tissue_type}: {specificity}")
                print(f"    F1 score for tissue {tissue_type}: {f1_score}")
                if np.isnan(sensitivity) or np.isnan(f1_score):
                    print(f"    Sensitivity or F1 for tissue {tissue_type} is not computable, setting to 0")
                    sensitivity, f1_score = 0.0, 0.0
            except Exception as e:
                print(f"    Error computing F1 Score for tissue {tissue_type}: {e}")
                sensitivity, specificity, f1_score = np.nan, np.nan, np.nan
            try:
                composite_score = calculate_composite_score(dice_score, hd95, f1_score, weights)
                print(f"    Composite score for tissue {tissue_type}: {composite_score}")
            except Exception as e:
                print(f"    Error computing Composite Score for tissue {tissue_type}: {e}")
                composite_score = np.nan

            patient_metrics.update({
                f"Dice_{tissue_type}": dice_score,
                f"HD95_{tissue_type}": hd95,
                f"Sensitivity_{tissue_type}": sensitivity,
                f"Specificity_{tissue_type}": specificity,
                f"F1_{tissue_type}": f1_score,
                f"Composite_Score_{tissue_type}": composite_score,
            })

        patient_scores.append(patient_metrics)
        print(f"Metrics for patient {patient_path[0]}: {patient_metrics}")

        print("Memory usage after patient computations:")
        print_memory_usage()

        # Memory cleanup
        del ground_truth, prob, seg, seg_out, batch_data
        gc.collect()
        torch.cuda.empty_cache()

        print("Memory usage after cleanup:")
        print_memory_usage()

        # Save intermediate results
        if idx % save_interval == 0:
            print("Memory usage before saving results:")
            print_memory_usage()
            print(f"Saving intermediate results at patient index {idx}")
            pd.DataFrame(patient_scores).to_csv(
                results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
            )
            print(f"Saved intermediate results to {results_save_path}")
            print("Memory usage after saving results:")
            print_memory_usage()
            
            patient_scores.clear()
            gc.collect()
            

# Final save if there are remaining entries in patient_scores
if patient_scores:
    pd.DataFrame(patient_scores).to_csv(
        results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
    )
    print(f"Saved final results to {results_save_path}")

print("Processing complete. All metrics saved.")

