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
        "vnet": lambda: models.vnet_model(),
        "attentionunet": lambda: models.attunet_model,
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
total_patients = len(config.val_loader)
confusion_metric = ConfusionMatrixMetric(
    metric_name=["sensitivity", "specificity", "f1 score"],
    include_background=False,
    compute_sample=False
)

# Initialize containers for ROC AUC computation per tissue type
y_true_dict = {label: [] for label in class_labels}
y_score_dict = {label: [] for label in class_labels}

# Intermediate result directory
intermediate_dir = os.path.join(config.output_dir, "intermediate_tensors")
roc_data_dir = os.path.join(config.output_dir, "roc_data")
os.makedirs(roc_data_dir, exist_ok=True)

roc_save_interval = 2
roc_save_path = os.path.join(roc_data_dir, "roc_data.pkl")
if not os.path.exists(roc_save_path):
    with open(roc_save_path, 'wb') as f:
        initial_data = {"y_true_dict": {label: [] for label in class_labels},
                        "y_score_dict": {label: [] for label in class_labels}}
        pickle.dump(initial_data, f)

os.makedirs(intermediate_dir, exist_ok=True)
tracemalloc.start()

print("Starting inference and metric computation")

save_interval=2
results_save_path = os.path.join(config.output_dir, f"patient_performance_scores_{config.model_name}.csv")

with torch.no_grad():
    for idx, batch_data in enumerate(config.val_loader):
        image = batch_data["image"].to(config.device)
        patient_path = batch_data["path"]

        print(f"Processing patient {idx}/{total_patients}: {patient_path[0]}")
        print("Memory usage at start of patient processing:")
        print_memory_usage()
        with autocast():
            prob = torch.sigmoid(model_inferer_test(image))
        del image
        # print(f"Probability output shape for patient {patient_path[0]}: {prob.shape}")
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        # print(f"Segmentation output for patient {patient_path[0]}: {seg.shape[0]} channels found.")
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4

        ground_truth = batch_data["label"][0].cpu().numpy()

        # Accumulate true labels and probabilities for each tissue type
        for channel, tissue_type in output_channel_to_class_label.items():
            tissue_true = (ground_truth == tissue_type).astype(np.int8)
            
            y_true_dict[tissue_type].extend(tissue_true)
            y_score_dict[tissue_type].extend(prob[0][channel].detach().cpu().numpy())

        # Add labels and probabilities for the background as well
        background_prob = 1 - np.max(prob[0].cpu().numpy(), axis=0)
        background_true = (ground_truth == 0).astype(np.int8)

        y_true_dict[0].extend(background_true)  
        y_score_dict[0].extend(background_prob)

        if idx % roc_save_interval == 0:
            print("Memory usage before saving ROC data:")
            print_memory_usage()

            # Load existing data from the pickle
            with open(roc_save_path, 'rb') as f:
                existing_data = pickle.load(f)
            
            # Append new data to the existing dictionaries
            for label in class_labels:
                existing_data["y_true_dict"][label].extend(y_true_dict[label])
                existing_data["y_score_dict"][label].extend(y_score_dict[label])

            # Save the updated data back to the same pickle file
            with open(roc_save_path, 'wb') as f:
                pickle.dump(existing_data, f, protocol=4)

            print(f"Updated ROC data saved to {roc_save_path}")

            # Clear dictionaries and reset for the next interval
            y_true_dict = {label: [] for label in class_labels}
            y_score_dict = {label: [] for label in class_labels}
            gc.collect()

            print("Memory usage after saving ROC data:")
            print_memory_usage()

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
        del ground_truth, prob, seg, seg_out
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

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')

            # print("[ Top 10 memory consuming lines ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            

# Final save if there are remaining entries in patient_scores
if patient_scores:
    pd.DataFrame(patient_scores).to_csv(
        results_save_path, index=False, mode='a', header=not os.path.exists(results_save_path)
    )
    print(f"Saved final results to {results_save_path}")

# Calculate and plot ROC AUC for each tissue type
print("Calculating and plotting ROC AUC per tissue type")

# Load the full ROC data from the single pickle file
roc_save_path = os.path.join(roc_data_dir, "roc_data.pkl")
with open(roc_save_path, 'rb') as f:
    roc_data = pickle.load(f)

# Separate the true labels and scores from the loaded data
full_y_true = roc_data["y_true_dict"]
full_y_score = roc_data["y_score_dict"]

# Calculate ROC AUC and plot for each tissue type
for tissue_type in class_labels:
    try:
        # Flatten arrays if needed
        y_true_flat = np.array(full_y_true[tissue_type]).ravel()
        y_score_flat = np.array(full_y_score[tissue_type]).ravel()
        
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true_flat, y_score_flat)
        print(f"ROC AUC for tissue {tissue_type}: {roc_auc}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Tissue {tissue_type} (AUC = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"ROC Curve for Tissue Type {tissue_type} ({config.model_name})")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(config.output_dir, f"roc_curve_tissue_{tissue_type}_{config.model_name}.png"))
        print(f"ROC curve for tissue {tissue_type} saved successfully")
    except Exception as e:
        print(f"Error calculating or plotting ROC AUC for tissue {tissue_type}: {e}")

print("Processing complete. All metrics saved.")

def mean_excluding_inf(values):
    finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
    return np.mean(finite_values) if finite_values else 0.0


