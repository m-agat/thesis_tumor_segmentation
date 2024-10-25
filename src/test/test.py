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

model = models.attunet_model
checkpoint = torch.load(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/attunet_model.pt"
)
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

def compute_dice_score_per_tissue(prediction, ground_truth, tissue_type):
    """
    Compute the Dice score for a specific tissue type.
    Args:
        prediction (numpy.ndarray): The predicted segmentation.
        ground_truth (numpy.ndarray): The ground truth segmentation.
        tissue_type (int): The label of the tissue type to evaluate (e.g., 1 for NCR, 2 for ED, 4 for ET).
    
    Returns:
        float: Dice score for the specified tissue type.
    """
    pred_tissue = (prediction == tissue_type).astype(np.float32)
    gt_tissue = (ground_truth == tissue_type).astype(np.float32)
    
    intersection = np.sum(pred_tissue * gt_tissue)
    union = np.sum(pred_tissue) + np.sum(gt_tissue)
    
    if union == 0:
        return 1.0  # If both prediction and ground truth have no pixels for this class, Dice is perfect.
    
    return (2.0 * intersection) / union

patient_scores = []
tissue_averages = {1: [], 2: [], 4: []}
with torch.no_grad():
    for batch_data in config.test_loader:
        image = batch_data["image"].cuda()
        patient_path = batch_data["path"]

        # Extract the affine matrix from the original image
        original_image_path = f"/home/agata/Desktop/thesis_tumor_segmentation/data/brats2021challenge/split/test/{patient_path[0]}/{patient_path[0]}_flair.nii.gz"
        original_nifti = nib.load(original_image_path)
        affine = original_nifti.affine

        # Perform inference
        prob = torch.sigmoid(model_inferer_test(image))
        seg = prob[0].detach().cpu().numpy()
        seg = (seg > 0.5).astype(np.int8)
        seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        seg_out[seg[1] == 1] = 2
        seg_out[seg[0] == 1] = 1
        seg_out[seg[2] == 1] = 4

        # Save the segmentation output as a NIfTI file using the extracted affine
        nifti_img = nib.Nifti1Image(seg_out, affine)
        nib.save(nifti_img, f"/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/AttentionUNet_Results_Test_Set/{patient_path[0]}_segmentation.nii.gz")

        ground_truth = batch_data["label"].cpu().numpy()  

        # Performance
        patient_dice = {"Patient": patient_path[0]}
        print('Patient: ', patient_dice["Patient"])
        for tissue_type in [1, 2, 4]:
            dice_score = compute_dice_score_per_tissue(seg_out, ground_truth, tissue_type)
            patient_dice[f"Dice_{tissue_type}"] = dice_score
            tissue_averages[tissue_type].append(dice_score)
            print(f'\t Dice score for {tissue_type}: {dice_score}')

        patient_scores.append(patient_dice)

        # Visualization
        slice_num = 90
        image_slice = (
            image[0, 0, slice_num].cpu().numpy()
        )  
        ground_truth_slice = (
            ground_truth[0, :, :, slice_num]
        )  
        predicted_slice = seg[0, :, :, slice_num]
        visualize_slices(
            image_slice, ground_truth_slice, predicted_slice, patient_path, slice_num
        )

# Convert individual scores to a DataFrame and save as CSV
df_patient_scores = pd.DataFrame(patient_scores)
df_patient_scores.to_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/patient_dice_scores.csv", index=False)

# Calculate average scores for each tissue type and save as a separate CSV
avg_scores = {
    "Tissue": ["NCR_(1)", "ED_(2)", "ET_(4)"],
    "Average Dice": [
        np.mean(tissue_averages[1]),
        np.mean(tissue_averages[2]),
        np.mean(tissue_averages[4])
    ]
}
df_avg_scores = pd.DataFrame(avg_scores)
df_avg_scores.to_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/average_dice_scores.csv", index=False)

print("Saved individual and average Dice scores.")