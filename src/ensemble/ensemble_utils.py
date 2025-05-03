import os, re, json
import torch, numpy as np, nibabel as nib, pandas as pd
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric, compute_hausdorff_distance
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.config as config

REGIONS = ["BG","NCR","ED","ET"]
DEVICE = config.device
ROI = config.roi
SW_BATCH = config.sw_batch_size
OVERLAP = config.infer_overlap
wts = {"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}

def load_model(model_cls, ckpt_path, device, roi, sw_batch, overlap):
    model = model_cls.to(device)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    inferer = partial(
        sliding_window_inference,
        roi_size=roi, sw_batch_size=sw_batch,
        predictor=model, overlap=overlap,
        mode="gaussian", sigma_scale=0.125,
        padding_mode="constant",
    )
    return model, inferer

def load_all_models(model_map, device, roi, sw_batch, overlap):
    return {
      name: load_model(cls, path, device, roi, sw_batch, overlap)
      for name,(cls,path) in model_map.items()
    }

def load_weights(path):
    with open(path) as f: return json.load(f)

def compute_composite_scores(metrics, wts):
    out = {}
    for r in REGIONS:
        hd = 1/(1+metrics[f"HD95 {r}"])
        out[r] = ( wts["Dice"] * metrics[f"Dice {r}"]
                  + wts["HD95"] * hd
                  + wts["Sensitivity"] * metrics[f"Sensitivity {r}"]
                  + wts["Specificity"] * metrics[f"Specificity {r}"] )
    return out

def normalize_weights(perf_weights):
    # perf_weights: { region: {model:score} }
    for r,mp in perf_weights.items():
        s = sum(mp.values())
        perf_weights[r] = {m: v/s for m,v in mp.items()}
    return perf_weights

def extract_patient_id(path):
    return re.findall(r"\d+", path)[-1]

def save_nifti(arr, ref_path, out_path, dtype=np.uint8):
    if isinstance(arr, torch.Tensor): arr = arr.detach().cpu().numpy()
    arr = arr.astype(dtype)
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), out_path)

def compute_metrics(pred_list, gt_list):
    """
    Compute Dice, HD95, Sensitivity, and Specificity for segmentation predictions.
    pred_list, gt_list: lists of length C, each tensor [H,W,D]
    """
    # Prepare batch-channel tensors
    y_pred = torch.stack([torch.as_tensor(p) for p in pred_list], dim=0).unsqueeze(0)
    y_gt   = torch.stack([torch.as_tensor(g) for g in gt_list],   dim=0).unsqueeze(0)

    # Dice (including background)
    dice_metric = DiceMetric(
        include_background=True,
        reduction=MetricReduction.NONE,
        get_not_nans=True,
    )
    dice_metric(y_pred=y_pred, y=y_gt)
    dice_scores, not_nans = dice_metric.aggregate()
    dice_scores = dice_scores.squeeze(0).cpu().numpy()  # shape: (C,)
    not_nans = not_nans.squeeze(0).cpu().numpy()    # shape: (C,)

    # Correct Dice when GT empty
    for i in range(len(dice_scores)):
        if not_nans[i] == 0:
            dice_scores[i] = 1.0 if pred_list[i].sum().item() == 0 else 0.0

    hd95 = np.zeros(len(dice_scores), dtype=float)
    for i in range(len(pred_list)):
        # prepare single-channel batch as [1,1,H,W,D]
        pred_ch = pred_list[i].unsqueeze(0).unsqueeze(0)
        gt_ch   = gt_list[i].unsqueeze(0).unsqueeze(0)
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

def save_metrics(metrics_list, out_dir):
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(out_dir,"patient_metrics.csv"),index=False)
    avg = df.drop(columns=["patient_id"]).mean().to_dict()
    with open(os.path.join(out_dir,"average_metrics.json"),"w") as f:
        json.dump(avg,f,indent=2)