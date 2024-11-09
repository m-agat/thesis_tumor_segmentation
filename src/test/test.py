import os
import sys
sys.path.append("../")
import models.models as models
import torch
import config.config as config
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
from utils.utils import visualize_slices
import nibabel as nib
import pandas as pd
from uncertainty.test_time_dropout import (
    test_time_dropout_inference,
    scale_to_range_0_100,
)
from uncertainty.test_time_augmentation import test_time_augmentation_inference
from utils.metrics import (
    compute_dice_score_per_tissue,
    compute_hd95,
    compute_sensitivity,
    calculate_composite_score,
)

base_path = "./outputs"
os.makedirs(base_path, exist_ok=True)
print(f"Output directory created: {base_path}")
model = models.segresnet_model
checkpoint = torch.load(config.model_file_path)
model.load_state_dict(checkpoint["state_dict"])
model.to(config.device)
model.eval()
model_inferer_test = partial(
    sliding_window_inference,
    roi_size=[config.roi[0], config.roi[1], config.roi[2]],
    sw_batch_size=1,
    predictor=model,
    overlap=0.6,
)

patient_scores = []
tissue_averages = {
    1: {"Dice": [], "HD95": [], "Sensitivity": [], "Composite_Score": []},
    2: {"Dice": [], "HD95": [], "Sensitivity": [], "Composite_Score": []},
    4: {"Dice": [], "HD95": [], "Sensitivity": [], "Composite_Score": []},
}
weights = {"Dice": 0.4, "HD95": 0.3, "Sensitivity": 0.3}
total_patients = len(config.test_loader)
with torch.no_grad():
    for idx, batch_data in enumerate(config.test_loader):
        image = batch_data["image"].cuda()
        patient_path = batch_data["path"]

        print(f"Processing patient {idx}/{total_patients}: {patient_path[0]}")

        # Extract the affine matrix from the original image
        original_image_path = os.path.join(
            config.test_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz"
        )
        original_nifti = nib.load(original_image_path)
        affine = original_nifti.affine
        header = original_nifti.header

        # Perform inference
        from torch.cuda.amp import autocast

        with autocast():
            prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4

        # Save the segmentation output as a NIfTI file using the extracted affine
        # nifti_img = nib.Nifti1Image(seg_out, affine)
        # nib.save(nifti_img, f"/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/VNet_Results_Test_Set/{patient_path[0]}_segmentation.nii.gz")

        ground_truth = batch_data["label"][0].cpu().numpy()

        # Performance
        patient_metrics = {"Patient": patient_path[0]}
        for tissue_type in [1, 2, 4]:
            dice_score = compute_dice_score_per_tissue(
                seg_out, ground_truth, tissue_type
            )
            hd95 = compute_hd95(seg_out, ground_truth, tissue_type)
            sensitivity = compute_sensitivity(seg_out, ground_truth, tissue_type)
            composite_score = calculate_composite_score(
                dice_score, hd95, sensitivity, weights
            )

            # Store metrics
            patient_metrics[f"Dice_{tissue_type}"] = dice_score
            patient_metrics[f"HD95_{tissue_type}"] = hd95
            patient_metrics[f"Sensitivity_{tissue_type}"] = sensitivity
            patient_metrics[f"Composite_Score_{tissue_type}"] = composite_score

            # Update tissue averages
            tissue_averages[tissue_type]["Dice"].append(dice_score)
            tissue_averages[tissue_type]["HD95"].append(hd95)
            tissue_averages[tissue_type]["Sensitivity"].append(sensitivity)
            tissue_averages[tissue_type]["Composite_Score"].append(composite_score)

        patient_scores.append(patient_metrics)
        print(patient_metrics)

        torch.cuda.empty_cache()

        # # Visualization
        # slice_num = 90
        # image_slice = (
        #     image[0, 0, slice_num].cpu().numpy()
        # )
        # ground_truth_slice = (
        #     ground_truth[:, :, slice_num]
        # )
        # predicted_slice = seg[0, :, :, slice_num]
        # visualize_slices(
        #     image_slice, ground_truth_slice, predicted_slice, patient_path, slice_num
        # )

df_patient_scores = pd.DataFrame(patient_scores)
df_patient_scores.to_csv(os.path.join(base_path, "patient_scores_swinunetr.csv"), index=False)

def mean_excluding_inf(values):
    finite_values = [v for v in values if not np.isinf(v)]
    return np.mean(finite_values) if finite_values else np.inf


avg_scores = {
    "Tissue": ["NCR_(1)", "ED_(2)", "ET_(4)"],
    "Average Dice": [
        np.mean(tissue_averages[1]["Dice"]),
        np.mean(tissue_averages[2]["Dice"]),
        np.mean(tissue_averages[4]["Dice"]),
    ],
    "Average HD95": [
        mean_excluding_inf(tissue_averages[1]["HD95"]),
        mean_excluding_inf(tissue_averages[2]["HD95"]),
        mean_excluding_inf(tissue_averages[4]["HD95"]),
    ],
    "Average Sensitivity": [
        np.mean(tissue_averages[1]["Sensitivity"]),
        np.mean(tissue_averages[2]["Sensitivity"]),
        np.mean(tissue_averages[4]["Sensitivity"]),
    ],
    "Average Composite Score": [
        np.mean(tissue_averages[1]["Composite_Score"]),
        np.mean(tissue_averages[2]["Composite_Score"]),
        np.mean(tissue_averages[4]["Composite_Score"]),
    ],
}
df_avg_scores = pd.DataFrame(avg_scores)
df_avg_scores.to_csv(os.path.join(base_path, "average_scores_swinunetr.csv"), index=False)

print("Saved individual and average metrics.")
