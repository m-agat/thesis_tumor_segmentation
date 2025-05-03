import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric, DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
import sys 

sys.path.append("../")
import config.config as config

DEVICE = config.device
def compute_metrics(pred_list, gt_list):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    pred_list, gt_list: lists of length C, each tensor [H,W,D]
    """
    # Prepare batch-channel tensors
    y_pred = torch.stack([torch.as_tensor(p).to(DEVICE) for p in pred_list], dim=0).unsqueeze(0)
    y_gt   = torch.stack([torch.as_tensor(g).to(DEVICE) for g in gt_list],   dim=0).unsqueeze(0)

    # Dice (including background)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.NONE,
        get_not_nans=True,
    )
    dice_metric(y_pred=y_pred, y=y_gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.squeeze(0).cpu().numpy()  # shape: (C,)
    not_nans     = not_nans.squeeze(0).cpu().numpy()    # shape: (C,)

    # Correct Dice when GT empty
    for i in range(len(dice_scores)):
        if not_nans[i] == 0:
            dice_scores[i] = 1.0 if pred_list[i].sum().item() == 0 else 0.0

    hd95 = np.zeros(len(dice_scores), dtype=float)
    for i in range(len(pred_list)):
        # prepare single-channel batch as [1,1,H,W,D]
        pred_ch = pred_list[i].unsqueeze(0).unsqueeze(0).to(DEVICE)
        gt_ch   = gt_list[i].unsqueeze(0).unsqueeze(0).to(DEVICE)

        pred_empty = torch.sum(pred_ch).item() == 0
        gt_empty   = not_nans[i] == 0
        
        if pred_empty and gt_empty:
            # both empty
            hd95[i] = 0.0
        elif gt_empty and not pred_empty:
            # GT absent, prediction present
            pred_array = pred_ch.cpu().numpy()[0,0]
            if pred_array.sum() > 0:
                com = center_of_mass(pred_array)
                if any(np.isnan(com)):
                    hd95[i] = 0.0
                else:
                    com_mask = np.zeros_like(pred_array, dtype=np.uint8)
                    coords = tuple(map(int, map(round, com)))
                    com_mask[coords] = 1
                    mask_tensor = torch.from_numpy(com_mask).unsqueeze(0).unsqueeze(0).to(torch.float32).to(DEVICE)
                    mock_hd = compute_hausdorff_distance(
                        y_pred=pred_ch,
                        y=mask_tensor,
                        include_background=False,
                        distance_metric="euclidean",
                        percentile=95,
                    )
                    hd95[i] = float(mock_hd.squeeze().item())
            else:
                hd95[i] = 0.0
        elif pred_empty and not gt_empty:
            # Prediction empty, GT present
            gt_array = gt_ch.cpu().numpy()[0,0]
            if gt_array.sum() > 0:
                com = center_of_mass(gt_array)
                if any(np.isnan(com)):
                    hd95[i] = 0.0
                else:
                    com_mask = np.zeros_like(gt_array, dtype=np.uint8)
                    coords = tuple(map(int, map(round, com)))
                    com_mask[coords] = 1
                    mask_tensor = torch.from_numpy(com_mask).unsqueeze(0).unsqueeze(0).to(torch.float32).to(DEVICE)
                    mock_hd = compute_hausdorff_distance(
                        y_pred=gt_ch,
                        y=mask_tensor,
                        include_background=False,
                        distance_metric="euclidean",
                        percentile=95,
                    )
                    hd95[i] = float(mock_hd.squeeze().item())
            else:
                hd95[i] = 0.0
        else:
            # both present: direct hd95
            hd_t = compute_hausdorff_distance(
                y_pred=pred_ch,
                y=gt_ch,
                include_background=False,
                distance_metric="euclidean",
                percentile=95,
            )
            hd95[i] = float(hd_t.squeeze().item())

    # Sensitivity / Specificity
    conf_metric = ConfusionMatrixMetric(
        include_background=True,
        metric_name=["sensitivity","specificity"],
        reduction="none",
    )
    conf_metric(y_pred=y_pred, y=y_gt)
    sens, spec = conf_metric.aggregate()
    sens = sens.squeeze(0).cpu().numpy()
    spec = spec.squeeze(0).cpu().numpy()

    # Correct sens/spec when GT empty
    for i in range(len(sens)):
        if not_nans[i] == 0:
            sens[i] = 1.0 if pred_list[i].sum().item() == 0 else 0.0
            spec[i] = 1.0

    return dice_scores, hd95, sens, spec