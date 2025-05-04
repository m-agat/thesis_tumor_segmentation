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
from torch.utils.data import Subset
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
# Ensure ROI matches image dimensions (3D)
# config.roi may be 2D; override to match volume depth if needed
def get_roi(img, default_roi):
    # img shape: (B,C,H,W,D) or (B,C,D,H,W)
    spatial = img.shape[-3:]
    # if default length mismatches, use spatial
    if len(default_roi) != len(spatial):
        return tuple(spatial)
    return default_roi

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
def compute_composite_scores(metrics, wts):
    scores = {}
    hd_bg = 1.0 / (1.0 + metrics['HD95 BG'])
    scores['BG'] = (wts['Dice'] * metrics['Dice BG'] + wts['HD95'] * hd_bg +
                    wts['Sensitivity'] * metrics['Sensitivity BG'] + wts['Specificity'] * metrics['Specificity BG'])
    for r in REGIONS[1:]:
        hd = 1.0 / (1.0 + metrics[f'HD95 {r}'])
        scores[r] = (wts['Dice'] * metrics[f'Dice {r}'] + wts['HD95'] * hd +
                     wts['Sensitivity'] * metrics[f'Sensitivity {r}'] + wts['Specificity'] * metrics[f'Specificity {r}'])
    return scores

# ----- I/O Helpers -----
def save_nifti(arr, ref_path, out_path, out_dtype=np.uint8):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = arr.astype(out_dtype)
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(arr, ref.affine, ref.header), out_path)
    print(f"Saved: {out_path}")

# ----- Calibration utility -----
def calibrate_temperature(fused_logits_list, labels_list):
    # Flatten and format
    logits_val = torch.cat([x.flatten(1) for x in fused_logits_list], dim=1).transpose(0,1)
    labels_val = torch.cat([y.flatten() for y in labels_list]).long()
    logits_val, labels_val = logits_val.to(DEVICE), labels_val.to(DEVICE)

    # fit single T
    T = torch.ones(1, requires_grad=True, device=DEVICE)
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=30)
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_val / T, labels_val)
        loss.backward()
        return loss
    optimizer.step(closure)
    return T.item()

# ----- Fusion logic -----
def run_fusion(img, models_dict, perf_weights, n_iter):
    preds = {}
    for name, (model, inferer) in models_dict.items():
        tta_m_np, tta_u_np = tta_variance(inferer, img, DEVICE, n_iterations=n_iter)
        tta_m = torch.as_tensor(tta_m_np, device=DEVICE).squeeze(0)
        tta_u = torch.as_tensor(tta_u_np, device=DEVICE).squeeze(0)
        inv_map = 1.0/(tta_u + 1e-6)
        preds[name] = tta_m * inv_map
    fused_logits = None
    for name, pred in preds.items():
        w_log = torch.stack([pred[i] * perf_weights[REGIONS[i]][name] for i in range(len(REGIONS))])
        fused_logits = w_log if fused_logits is None else fused_logits + w_log
    return fused_logits

# ----- Ensemble + calibration -----
def ensemble_calibrate(loader, models_dict, weights, n_calib=20, n_iter=10):
    # compute perf_weights once
    perf_weights = {r: {} for r in REGIONS}
    for m in models_dict:
        metrics = json.load(open(f"../models/performance/{m}/average_metrics.json"))
        cs = compute_composite_scores(metrics, weights)
        for r in REGIONS:
            perf_weights[r][m] = cs[r]
    for r in REGIONS:
        tot = sum(perf_weights[r].values())
        for m in perf_weights[r]: perf_weights[r][m] /= tot

    # calibrate on first n_calib batches from loader
    fused_list, label_list = [], []
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for i, batch in enumerate(loader):
            if i >= n_calib:
                break
            img = batch['image'].to(DEVICE)
            gt  = batch['label'].to(DEVICE)
            print("gt.shape", gt.shape)
            fused_logits = run_fusion(img, models_dict, perf_weights, n_iter)
            fused_list.append(fused_logits.cpu())
            label_idx = gt.argmax(dim=1)  # shape [B,H,W,D]
            label_list.append(label_idx.cpu())
    T_opt = calibrate_temperature(fused_list, label_list)
    print("Calibrated T =", T_opt)
    return T_opt

if __name__ == "__main__":
    models_dict = load_all_models()
    weights = {"Dice":0.45, "HD95":0.15, "Sensitivity":0.3, "Specificity":0.1}
    _, val_loader = dataloaders.get_loaders(batch_size=config.batch_size,
                                            json_path=config.json_path,
                                            basedir=config.root_dir,
                                            fold=None,
                                            roi=config.roi,
                                            use_final_split=True)
    T_opt = ensemble_calibrate(val_loader, models_dict, weights, n_calib=1, n_iter=10)
