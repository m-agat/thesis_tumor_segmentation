import os
import sys
import torch
import numpy as np
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
import re
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass

sys.path.append("../")
import models.models as models
import config.config as config
from dataset import dataloaders

def compute_metrics(pred, gt):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    """
    dice_metric = DiceMetric(
        include_background=False, reduction=MetricReduction.NONE, get_not_nans=True
    )
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,
        metric_name=["sensitivity", "specificity"],
        reduction="none",
        compute_sample=False,
    )

    # Convert MetaTensors to plain tensors if needed.
    pred = [p.detach().clone() if hasattr(p, "detach") else p for p in pred]
    gt = [g.detach().clone() if hasattr(g, "detach") else g for g in gt]

    pred_stack = torch.stack(pred)
    gt_stack = torch.stack(gt)

    # Compute Dice Scores
    dice_metric(y_pred=pred, y=gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.cpu().numpy()

    for i, dice_score in enumerate(dice_scores):
        if not_nans[i] == 0:  # Tissue is absent in ground truth
            pred_empty = torch.sum(pred_stack[i]).item() == 0
            dice_scores[i] = 1.0 if pred_empty else 0.0

    # Compute HD95
    hd95 = compute_hausdorff_distance(
        y_pred=pred_stack,
        y=gt_stack,
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
    )
    hd95 = hd95.squeeze(0).cpu().numpy()
    for i in range(len(hd95)):
        # Use the i-th class mask directly.
        pred_empty = torch.sum(pred[i]).item() == 0
        gt_empty = not_nans[i] == 0

        if pred_empty and gt_empty:
            print(f"Region {i}: Both GT and Prediction are empty. Setting HD95 to 0.")
            hd95[i] = 0.0

        elif gt_empty and not pred_empty:  # Ground truth is absent.
            pred_array = pred[i].cpu().numpy()  # Use pred[i] directly.
            if np.sum(pred_array) > 0:
                # Compute Center of Mass for the predicted mask
                com = center_of_mass(pred_array)
                com_mask = np.zeros_like(pred_array, dtype=np.uint8)
                com_coords = tuple(
                    map(int, map(round, com))
                )  # Round and convert to integer indices
                com_mask[com_coords] = 1

                # Convert CoM mask back to tensor
                com_mask_tensor = (
                    torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                )

                # Compute Hausdorff Distance between prediction and CoM mask
                mock_val = compute_hausdorff_distance(
                    y_pred=torch.stack(pred)[i].unsqueeze(0),
                    y=com_mask_tensor.unsqueeze(0),
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95,
                )

                print(f"Mock HD95 for region {i} (GT absent):", mock_val.item())
                print(f"Before update, hd95: {hd95}")
                hd95[i] = mock_val.item()
                print(f"After update, hd95: {hd95}")
            else:
                # No prediction or GT; HD95 = 0
                hd95[i] = 0.0

        elif pred_empty and not gt_empty:  # Model predicts tissue is absent
            gt_array = torch.stack(gt)[i].cpu().numpy()
            if np.sum(gt_array) > 0:
                # Compute Center of Mass for the GT mask
                com = center_of_mass(gt_array)
                com_mask = np.zeros_like(gt_array, dtype=np.uint8)
                com_coords = tuple(
                    map(int, map(round, com))
                )  # Round and convert to integer indices
                com_mask[com_coords] = 1

                # Convert CoM mask back to tensor
                com_mask_tensor = (
                    torch.from_numpy(com_mask).to(torch.float32).to(config.device)
                )

                # Compute Hausdorff Distance between GT CoM and empty prediction
                mock_val = compute_hausdorff_distance(
                    y_pred=torch.stack(gt)[i].unsqueeze(0),
                    y=com_mask_tensor.unsqueeze(0),
                    include_background=False,
                    distance_metric="euclidean",
                    percentile=95,
                )

                print(
                    f"Mock HD95 for region {i} (Prediction absent):",
                    mock_val.item(),
                )
                print(f"Before update, hd95: {hd95}")
                hd95[i] = mock_val.item()
                print(f"After update, hd95: {hd95}")
            else:
                print(f"Warning: GT mask for region {i} is unexpectedly empty.")
                hd95[i] = 0.0

    # Compute Sensitivity & Specificity
    confusion_metric(y_pred=pred, y=gt)
    sensitivity, specificity = confusion_metric.aggregate()
    sensitivity = sensitivity.squeeze(0).cpu().numpy()
    specificity = specificity.squeeze(0).cpu().numpy()

    for i in range(len(sensitivity)):
        if not_nans[i] == 0:  # Tissue is absent
            pred_empty = torch.sum(pred_stack[i]).item() == 0
            sensitivity[i] = 1.0 if pred_empty else 0.0
            specificity[i] = 1.0

    return dice_scores, hd95, sensitivity, specificity

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


def save_segmentation_as_nifti(predicted_segmentation, reference_image_path, output_path):
    """
    Save the predicted segmentation as a NIfTI file.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()

    predicted_segmentation = predicted_segmentation.astype(np.uint8)

    ref_img = nib.load(reference_image_path)
    seg_img = nib.Nifti1Image(predicted_segmentation, affine=ref_img.affine, header=ref_img.header)
    nib.save(seg_img, output_path)

    print(f"Segmentation saved to {output_path}")


def extract_patient_id(path):
    """
    Extract patient ID from the file path.
    """
    numbers = re.findall(r"\d+", path)
    patient_id = numbers[-1]
    return patient_id


def ensemble_segmentation(test_loader, models_dict, patient_id=None, output_dir="../models/predictions", model_name="model"):
    """
    Perform segmentation using a single model.
    
    Parameters:
    - patient_id: ID of the patient whose scan is being segmented.
    - test_loader: Dataloader for the test set.
    - models_dict: Dictionary containing a single trained model and its inferer.
    - output_dir: Directory where segmentations will be saved.
    - model_name: Name of the model used (for output file naming).
    """
    os.makedirs(output_dir, exist_ok=True)
    if patient_id is not None:
        test_data_loader = config.find_patient_by_id(patient_id, test_loader)
    else:
        test_data_loader = test_loader

    with torch.no_grad():
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            gt = batch_data["label"].to(config.device)
            reference_image_path = batch_data["path"][0]
            patient_id = extract_patient_id(reference_image_path)
            model, inferer = next(iter(models_dict.values()))
            print(f"\nProcessing patient: {patient_id} using {model_name}\n")

            logits = inferer(image).squeeze(0)
            seg = torch.nn.functional.softmax(logits, dim=0).argmax(dim=0).unsqueeze(0)

            pred_one_hot = [(seg == i).float() for i in range(0, 4)]
            if gt.shape[1] == 4:
                # Ground truth is already one-hot encoded (assume channel 0 is background).
                # Extract channels 1,2,3 and permute to have shape [3, 1, H, W, D].
                gt_one_hot = gt.permute(1, 0, 2, 3, 4)
            else:
                # Ground truth is not one-hot encoded, so create one-hot encoding.
                gt_one_hot = [(gt == i).float() for i in range(0, 4)]
                gt_one_hot = torch.stack(gt_one_hot)

            # Get performance metrics
            dice, hd95, sensitivity, specificity = compute_metrics(
                pred_one_hot, gt_one_hot
            )
            print(
                f"Dice BG: {dice[0].item():.4f}, Dice NCR: {dice[1].item():.4f}, Dice ED: {dice[2].item():.4f}, Dice ET: {dice[3].item():.4f}\n"
                f"HD95 BG: {hd95[0].item():.2f}, HD95 NCR: {hd95[1].item():.2f}, HD95 ED: {hd95[2].item():.2f}, HD95 ET: {hd95[3].item():.2f}\n"
                f"Sensitivity BG: {sensitivity[0].item():.4f}, NCR: {sensitivity[1].item():.4f}, ED: {sensitivity[2].item():.4f}, ET: {sensitivity[3].item():.4f}\n"
                f"Specificity BG: {specificity[0].item():.4f}, NCR: {specificity[1].item():.4f}, ED: {specificity[2].item():.4f}, ET: {specificity[3].item():.4f}\n"
            )

            seg = seg.squeeze(0)

            output_path = os.path.join(output_dir, f"{model_name}_{patient_id}_pred_seg.nii.gz")
            save_segmentation_as_nifti(seg, reference_image_path, output_path)
            torch.cuda.empty_cache()
            # Process only one patient for this example.
            break


if __name__ == "__main__":
    # Change this variable to the model you want to use:
    selected_model = "vnet"  # Options: "attunet", "swinunetr", "segresnet"

    # Map model names to their corresponding model class and checkpoint path.
    model_config = {
        "attunet": (models.attunet_model, config.model_paths["attunet"]),
        "swinunetr": (models.swinunetr_model, config.model_paths["swinunetr"]),
        "segresnet": (models.segresnet_model, config.model_paths["segresnet"]),
        "vnet": (models.vnet_model, config.model_paths["vnet"]),
    }

    if selected_model not in model_config:
        raise ValueError(f"Selected model '{selected_model}' is not recognized. Choose from: {list(model_config.keys())}")

    model_class, checkpoint_path = model_config[selected_model]
    loaded_model = load_model(model_class, checkpoint_path, config.device)
    models_dict = {selected_model: loaded_model}

    # Define the output directory based on the selected model.
    output_dir = os.path.join("../models/predictions", selected_model)

    # Change the patient_id here if necessary.
    patient_id = "01531"
    train_loader, _ = dataloaders.get_loaders(
        batch_size=config.batch_size,
        json_path=config.json_path,
        basedir=config.root_dir,
        fold=None,
        roi=config.roi,
        use_final_split=True,
    )

    ensemble_segmentation(train_loader, models_dict, patient_id=patient_id, output_dir=output_dir, model_name=selected_model)
