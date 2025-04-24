import os
import re
import json
import torch
import numpy as np
import nibabel as nib
import pandas as pd
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric, DiceMetric
from monai.utils.enums import MetricReduction
from scipy.ndimage import center_of_mass
from torch.amp import autocast

# project imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.models as models
import config.config as config
from uncertainty.test_time_augmentation import tta_variance
from uncertainty.test_time_dropout import ttd_variance, minmax_uncertainties
from dataset import dataloaders

# constants
REGIONS = ["BG", "NCR", "ED", "ET"]
DEVICE = config.device
ROI = config.roi
SW_BATCH = config.sw_batch_size
OVERLAP = config.infer_overlap

# ----- Model Loading -----
def load_model(model_class, checkpoint_path):
    model = model_class.to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    inferer = partial(
        sliding_window_inference,
        roi_size=ROI,
        sw_batch_size=SW_BATCH,
        predictor=model,
        overlap=OVERLAP,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="constant",
    )
    return model, inferer


def load_all_models():
    return {
        name: load_model(cfg_model, cfg_path)
        for name, (cfg_model, cfg_path) in {
            'swinunetr': (models.swinunetr_model, config.model_paths['swinunetr']),
            'segresnet': (models.segresnet_model, config.model_paths['segresnet']),
            'attunet':    (models.attunet_model,   config.model_paths['attunet']),
        }.items()
    }

# ----- Weights & Scoring -----
def load_weights(path):
    with open(path) as f:
        return json.load(f)


def compute_composite_scores(metrics, wts):
    scores = {}
    # background
    hd_bg = 1.0 / (1.0 + metrics['HD95 BG'])
    scores['BG'] = (wts['Dice'] * metrics['Dice BG'] +
                    wts['HD95'] * hd_bg +
                    wts['Sensitivity'] * metrics['Sensitivity BG'] +
                    wts['Specificity'] * metrics['Specificity BG'])
    # tumor regions
    for r in REGIONS[1:]:
        hd = 1.0 / (1.0 + metrics[f'HD95 {r}'])
        scores[r] = (wts['Dice'] * metrics[f'Dice {r}'] +
                     wts['HD95'] * hd +
                     wts['Sensitivity'] * metrics[f'Sensitivity {r}'] +
                     wts['Specificity'] * metrics[f'Specificity {r}'])
    return scores

# ----- I/O Helpers -----
def save_nifti(arr, ref_path, out_path, out_dtype=np.uint8):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = arr.astype(out_dtype)
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), out_path)
    print(f"Saved: {out_path}")

# ----- Metrics Computation -----
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
    not_nans     = not_nans.squeeze(0).cpu().numpy()    # shape: (C,)

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

# ----- Ensemble Inference -----
def ensemble_segmentation(loader, models_dict, wts, n_iter=10, pid=None, out_dir="./output"):
    os.makedirs(out_dir, exist_ok=True)
    perf_weights = {r: {} for r in REGIONS}
    for m in models_dict:
        metrics = load_weights(f"../models/performance/{m}/average_metrics.json")
        cs = compute_composite_scores(metrics, wts)
        for r in REGIONS:
            perf_weights[r][m] = cs[r]
    for r in REGIONS:
        total = sum(perf_weights[r].values())
        for m in perf_weights[r]:
            perf_weights[r][m] /= total

    patient_metrics = []
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        data_loader = config.find_patient_by_id(pid, loader) if pid else loader
        for batch in data_loader:
            img = batch['image'].to(DEVICE)
            ref_path = batch['path'][0]
            gt = batch['label'].to(DEVICE)
            pid_str = re.findall(r"\d+", ref_path)[-1]
            print(f"Processing {pid_str}")

            preds, uncs = {}, {}
            for name, (model, inferer) in models_dict.items():
                tta_m_np, tta_u_np = tta_variance(inferer, img, DEVICE, n_iterations=n_iter)
                tta_m = torch.as_tensor(tta_m_np, device=DEVICE).squeeze(0)
                tta_u = torch.as_tensor(tta_u_np, device=DEVICE).squeeze(0)
                inv_map = 1.0/(tta_u + 1e-6)
                preds[name] = tta_m * inv_map
                uncs[name] = tta_u

            fused_logits = None
            for name, pred in preds.items():
                w_log  = torch.stack([pred[i] * perf_weights[REGIONS[i]][name] for i in range(len(REGIONS))])
                fused_logits = w_log  if fused_logits is None else fused_logits +  w_log
            
            T_opt = 4.117959976196289
            probabilities = torch.softmax(fused_logits/T_opt, dim=0)
            seg = probabilities.argmax(dim=0)

            fused_unc = {}
            for idx, region in enumerate(REGIONS[1:], start=1):
                acc = None
                for name in uncs:
                    w = perf_weights[region][name]
                    reg_unc = uncs[name][idx]
                    acc = reg_unc * w if acc is None else acc + reg_unc * w
                fused_unc[region] = minmax_uncertainties(acc.cpu().numpy())

            pred_list = [(seg == i).float() for i in range(len(REGIONS))]
            if gt.shape[1] == len(REGIONS):
                gt_list = [gt[:, i].squeeze(0) for i in range(len(REGIONS))]
            else:
                gt_list = [(gt == i).float().squeeze(0) for i in range(len(REGIONS))]

            # Compute performance metrics
            dice_scores, hd95, sens, spec = compute_metrics(pred_list, gt_list)
            patient_metrics.append({
                'patient_id': pid_str,
                **{f"Dice {REGIONS[i]}": dice_scores[i] for i in range(1, len(REGIONS))},
                **{f"HD95 {REGIONS[i]}": hd95[i] for i in range(1, len(REGIONS))},
                **{f"Sensitivity {REGIONS[i]}": sens[i] for i in range(1, len(REGIONS))},
                **{f"Specificity {REGIONS[i]}": spec[i] for i in range(1, len(REGIONS))},
                'Dice overall': float(np.mean(dice_scores[1:])),
                'HD95 overall': float(np.mean(hd95[1:])),
                'Sensitivity overall': float(np.mean(sens[1:])),
                'Specificity overall': float(np.mean(spec[1:])),
            })

            print("Dice: ", dice_scores)
            print("HD95: ", hd95)
            print("Sensitivity: ", sens)
            print("Specificity: ", spec)

            save_nifti(probabilities.cpu(), ref_path, os.path.join(out_dir, f"softmax_{pid_str}.nii.gz"), out_dtype=np.float32)
            save_nifti(seg.cpu(), ref_path, os.path.join(out_dir, f"seg_{pid_str}.nii.gz"))
            for region, unc_map in fused_unc.items():
                save_nifti(unc_map, ref_path, os.path.join(out_dir, f"uncertainty_{region}_{pid_str}.nii.gz"), out_dtype=np.float32)

    df = pd.DataFrame(patient_metrics)
    df.to_csv(os.path.join(out_dir, "patient_metrics.csv"), index=False)
    avg = df.drop(columns=['patient_id']).mean().to_dict()
    with open(os.path.join(out_dir, "average_metrics.json"), 'w') as f:
        json.dump(avg, f, indent=2)
    print("Ensemble inference complete.")

if __name__ == "__main__":
    models_dict = load_all_models()
    weights = {"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}
    test_loader = dataloaders.load_test_data(config.json_path, config.root_dir)
    ensemble_segmentation(test_loader, models_dict, weights,
                          n_iter=10, pid=None, out_dir="./output/tta")
