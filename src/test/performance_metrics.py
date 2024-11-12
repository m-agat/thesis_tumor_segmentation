import os
import sys
import torch
import gc
import config.config as config
import models.models as models
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_sensitivity,
    compute_specificity,
    calculate_composite_score,
    compute_f1_score,
)

# Set base output directory
base_path = config.output_dir
os.makedirs(base_path, exist_ok=True)
print(f"Output directory: {base_path}")

# Load model function
def load_model():
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
    return model

model = load_model()
model_inferer_test = partial(
    sliding_window_inference,
    roi_size=config.roi,
    sw_batch_size=config.sw_batch_size,
    predictor=model,
    overlap=config.infer_overlap,
)

weights = {"Dice": 0.4, "HD95": 0.4, "F1": 0.2}
patient_scores = []
tissue_averages = {
    1: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
    2: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
    4: {"Dice": [], "HD95": [], "Sensitivity": [], "Specificity": [], "F1": [], "Composite_Score": []},
}
tissue_roc_data = {1: {"true": [], "pred": []}, 2: {"true": [], "pred": []}, 4: {"true": [], "pred": []}}
total_patients = len(config.val_loader)

# Inference loop
with torch.no_grad():
    for idx, batch_data in enumerate(config.val_loader):
        try:
            image = batch_data["image"].to(config.device)
            patient_path = batch_data["path"]

            print(f"Processing patient {idx}/{total_patients}: {patient_path[0]}")
            with torch.cuda.amp.autocast():
                prob = torch.sigmoid(model_inferer_test(image))
            seg = (prob[0] > 0.5).cpu().numpy().astype(np.int8)
            
            seg_out = np.zeros(seg.shape[1:], dtype=np.int8)
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4

            ground_truth = batch_data["label"][0].cpu().numpy().astype(np.float16)

            patient_metrics = {"Patient": patient_path[0]}
            for tissue_type in [1, 2, 4]:
                gt_mask = (ground_truth == tissue_type).astype(np.float16)
                pred_mask = (seg_out == tissue_type).astype(np.float16)

                # Aggregate ROC data
                tissue_roc_data[tissue_type]["true"].extend(gt_mask.ravel())
                tissue_roc_data[tissue_type]["pred"].extend(pred_mask.ravel())

                # Compute and store metrics
                dice_score = compute_dice_score_per_tissue(seg_out, ground_truth, tissue_type)
                hd95 = compute_hd95(seg_out, ground_truth, tissue_type)
                sensitivity = compute_sensitivity(seg_out, ground_truth, tissue_type)
                specificity = compute_specificity(seg_out, ground_truth, tissue_type)
                f1_score = compute_f1_score(seg_out, ground_truth, tissue_type)
                composite_score = calculate_composite_score(dice_score, hd95, f1_score, weights)

                patient_metrics.update({
                    f"Dice_{tissue_type}": dice_score,
                    f"HD95_{tissue_type}": hd95,
                    f"Sensitivity_{tissue_type}": sensitivity,
                    f"Specificity_{tissue_type}": specificity,
                    f"F1_{tissue_type}": f1_score,
                    f"Composite_Score_{tissue_type}": composite_score,
                })

                # Accumulate metrics for averaging
                tissue_averages[tissue_type]["Dice"].append(dice_score)
                tissue_averages[tissue_type]["HD95"].append(hd95)
                tissue_averages[tissue_type]["Sensitivity"].append(sensitivity)
                tissue_averages[tissue_type]["Specificity"].append(specificity)
                tissue_averages[tissue_type]["F1"].append(f1_score)
                tissue_averages[tissue_type]["Composite_Score"].append(composite_score)

            patient_scores.append(patient_metrics)
            print(patient_metrics)

            # Memory cleanup
            del image, ground_truth, prob, seg, seg_out
            gc.collect()
            torch.cuda.empty_cache()

            # Save intermediate results
            if idx % 10 == 0:
                pd.DataFrame(patient_scores).to_csv(
                    os.path.join(base_path, f"partial_patient_scores_{config.model_name}_{idx}.csv"), index=False,
                )

        except Exception as e:
            print(f"Error processing patient {patient_path[0]}: {e}")
            continue

# Final save for all patient scores
pd.DataFrame(patient_scores).to_csv(
    os.path.join(base_path, f"patient_scores_{config.model_name}.csv"), index=False
)

# Compute and plot ROC curves for each tissue type
for tissue_type in [1, 2, 4]:
    fpr, tpr, _ = roc_curve(
        tissue_roc_data[tissue_type]["true"], tissue_roc_data[tissue_type]["pred"]
    )
    auc = roc_auc_score(
        tissue_roc_data[tissue_type]["true"], tissue_roc_data[tissue_type]["pred"]
    )
    plt.plot(fpr, tpr, label=f"Tissue {tissue_type} (AUC = {auc:.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Each Tissue Type")
plt.legend(loc="lower right")
plt.savefig(os.path.join(base_path, f"tissue_level_roc_curves_{config.model_name}.png"))

# Model-level ROC and AUC
all_true = np.concatenate([tissue_roc_data[t]["true"] for t in [1, 2, 4]])
all_pred = np.concatenate([tissue_roc_data[t]["pred"] for t in [1, 2, 4]])
fpr, tpr, _ = roc_curve(all_true, all_pred)
model_auc = roc_auc_score(all_true, all_pred)

plt.figure()
plt.plot(fpr, tpr, label=f"Model Level (AUC = {model_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Model-Level ROC Curve ({config.model_name})")
plt.legend(loc="lower right")
plt.savefig(os.path.join(base_path, f"model_level_roc_curve_{config.model_name}.png"))

def mean_excluding_inf(values):
    finite_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
    return np.mean(finite_values) if finite_values else 0.0

# Final average scores
avg_scores = {
    "Tissue": ["1", "2", "4"],
    "Average Dice": [mean_excluding_inf(tissue_averages[t]["Dice"]) for t in [1, 2, 4]],
    "Average HD95": [mean_excluding_inf(tissue_averages[t]["HD95"]) for t in [1, 2, 4]],
    "Average Sensitivity": [
        mean_excluding_inf(tissue_averages[t]["Sensitivity"]) for t in [1, 2, 4]
    ],
    "Average Specificity": [
        mean_excluding_inf(tissue_averages[t]["Specificity"]) for t in [1, 2, 4]
    ],
    "Average F1": [mean_excluding_inf(tissue_averages[t]["F1"]) for t in [1, 2, 4]],
    "Average Composite Score": [
        mean_excluding_inf(tissue_averages[t]["Composite_Score"]) for t in [1, 2, 4]
    ],
}
pd.DataFrame(avg_scores).to_csv(
    os.path.join(base_path)
)

print("Saved individual and average metrics.")
