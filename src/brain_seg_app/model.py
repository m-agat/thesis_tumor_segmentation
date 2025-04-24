import os
import sys
import torch
import numpy as np
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time 

# Add custom modules
sys.path.append("../")
import models.models as models
import config.config as config
from uncertainty.test_time_augmentation import tta_variance
from uncertainty.test_time_dropout import ttd_variance, minmax_uncertainties
from dataset import dataloaders

REGIONS = ["BG", "NCR", "ED", "ET"]
DEVICE = config.device
ROI = config.roi
SW_BATCH = config.sw_batch_size
OVERLAP = config.infer_overlap

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
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="constant",
    )

def load_all_models():
    """
    Load all segmentation models into a dictionary.
    """
    return {
        "swinunetr": load_model(models.swinunetr_model, config.model_paths["swinunetr"], config.device),
        "segresnet": load_model(models.segresnet_model, config.model_paths["segresnet"], config.device),
        "attunet": load_model(models.attunet_model, config.model_paths["attunet"], config.device),
    }

def load_weights(performance_weights_path):
    with open(performance_weights_path) as f:
        return json.load(f)

def compute_composite_scores(metrics, weights):
    """Compute weighted composite scores for a model."""
    composite_scores = {}
    normalized_hd95_bg = 1 / (1 + metrics["HD95 BG"])
    composite_scores["BG"] = (
        weights["Dice"] * metrics["Dice BG"]
        + weights["HD95"] * normalized_hd95_bg
        + weights["Sensitivity"] * metrics["Sensitivity BG"]
        + weights["Specificity"] * metrics["Specificity BG"]
    )
    for region in ["NCR", "ED", "ET"]:
        normalized_hd95 = 1 / (1 + metrics[f"HD95 {region}"])
        composite_scores[region] = (
            weights["Dice"] * metrics[f"Dice {region}"]
            + weights["HD95"] * normalized_hd95
            + weights["Sensitivity"] * metrics[f"Sensitivity {region}"]
            + weights["Specificity"] * metrics[f"Specificity {region}"]
        )
    return composite_scores

def save_segmentation_as_nifti(predicted_segmentation, ref_img, output_path):
    """
    Save the predicted segmentation as a NIfTI file.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()
    predicted_segmentation = predicted_segmentation.astype(np.uint8)
    seg_img = nib.Nifti1Image(predicted_segmentation, affine=ref_img.affine, header=ref_img.header)
    nib.save(seg_img, output_path)
    print(f"Segmentation saved to {output_path}")

def save_probability_map_as_nifti(prob_map, ref_img, output_path):
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
    prob_map = prob_map.astype(np.float32)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    prob_img = nib.Nifti1Image(prob_map, affine=ref_img.affine, header=hdr)
    nib.save(prob_img, output_path)
    print(f"Probability map saved to {output_path}")

def extract_patient_id(path):
    """
    Extracts the patient ID from a filename by capturing tokens that consist solely
    of digits or uppercase letters, while ignoring tokens that are exactly "NCR", "ED", or "ET".
    
    Examples:
      "preproc_ARE_flair.nii"         -> returns "ARE"
      "uncertainty_NCR_00657.nii.gz"    -> returns "00657"
      "preproc_NCR_ABC.nii"             -> returns "ABC"
    """
    fname = os.path.basename(path)
    # Capture tokens that follow an underscore and consist of digits and/or uppercase letters.
    tokens = re.findall(r'_([\dA-Z]+)', fname)
    
    # Define tokens to ignore.
    ignore = {"NCR", "ED", "ET"}
    
    # Filter tokens: Allow numeric tokens or alphabetic tokens not in ignore.
    valid_tokens = []
    for token in tokens:
        if token.isdigit():
            valid_tokens.append(token)
        elif token.isalpha() and token not in ignore:
            valid_tokens.append(token)
    
    # Return the last valid token if available (you could also choose the first, depending on your naming scheme).
    if valid_tokens:
        return valid_tokens[-1]
    return "UNKNOWN"

def save_uncertainty_as_nifti(uncertainty_map, ref_img, output_path):
    """
    Save a 3D uncertainty map as a NIfTI file with optional scaling.
    """
    if isinstance(uncertainty_map, torch.Tensor):
        uncertainty_map = uncertainty_map.cpu().numpy()
    uncertainty_map = minmax_uncertainties(uncertainty_map)
    uncertainty_map = uncertainty_map.astype(np.float32)
    uncertainty_nifti = nib.Nifti1Image(uncertainty_map, affine=ref_img.affine, header=ref_img.header)
    nib.save(uncertainty_nifti, output_path)
    print(f"Uncertainty map saved to {output_path}")

def make_progress_callback(offset, total, progress_queue, scale=1.0):
    """
    Returns a callback that multiplies the reported current iteration by 'scale'.
    """
    def callback(current, _):
        # Multiply the current iteration count by the scale factor.
        overall_pct = int((offset + current * scale) / total * 100)
        progress_queue.put(overall_pct)
    return callback

def compute_model_inference(args, progress_queue):
    model_name, model, inferer, image, device, n_iterations, w_ttd, w_tta = args
    # Create a separate CUDA stream for this thread
    stream = torch.cuda.Stream(device=device)
    total_iterations = n_iterations * 2  # n_iterations for TTD and n_iterations for TTA for 4 threads

    with torch.cuda.stream(stream):
        # For TTD, progress starts at 0
        ttd_mean, ttd_uncertainty = ttd_variance(
            model, inferer, image, device,
            n_iterations=n_iterations,
            on_step=make_progress_callback(0, total_iterations, progress_queue, scale=0.5)
        )
        # For TTA, progress starts after TTD is done
        tta_mean, tta_uncertainty = tta_variance(
            inferer, image, device,
            n_iterations=n_iterations,
            on_step=make_progress_callback(n_iterations, total_iterations, progress_queue, scale=0.5)
        )

        eps = 1e-1 # to avoid division by zero
        # Compute voxel-level inverse uncertainty maps
        inv_ttd = 1.0 / np.clip(ttd_uncertainty, a_min=eps, a_max=None)
        inv_tta = 1.0 / np.clip(tta_uncertainty, a_min=eps, a_max=None)

        # Use ttd_mean as base and weight with inverse uncertainties
        adjusted_prediction = ttd_mean * inv_ttd
        adjusted_prediction = adjusted_prediction * inv_tta
        adjusted_prediction = np.squeeze(adjusted_prediction)

        # Compute combined uncertainty for the model
        combined_uncertainty = (ttd_uncertainty + tta_uncertainty) / 2.0
        combined_uncertainty = np.squeeze(combined_uncertainty)
    
    torch.cuda.synchronize()
    pred_t = torch.from_numpy(adjusted_prediction).to(device, torch.float32)
    unc_t  = torch.from_numpy(combined_uncertainty).to(device, torch.float32)
    return model_name, pred_t, unc_t

def update_progress_bar(progress_bar, progress_queue, last_progress):
    """Poll the progress queue and update the progress bar only if progress increases."""
    while not progress_queue.empty():
        pct = progress_queue.get()
        # Only update if the new percentage is greater than the last recorded value.
        if pct > last_progress[0]:
            last_progress[0] = pct
            progress_bar.progress(pct, text=f"{pct}% completed")

def ensemble_segmentation(test_loader, models_dict, composite_score_weights, n_iterations=10, patient_id=None, output_dir="./assets/segmentations", progress_bar=None):
    os.makedirs(output_dir, exist_ok=True)
    test_data_loader = config.find_patient_by_id(patient_id, test_loader) if patient_id is not None else test_loader

    # Get performance weights for all classes (BG, NCR, ED, ET)
    perf_weights = {r: {} for r in REGIONS}
    for m in models_dict:
        metrics = load_weights(f"../models/performance/{m}/average_metrics.json")
        cs = compute_composite_scores(metrics, composite_score_weights)
        for r in REGIONS:
            perf_weights[r][m] = cs[r]
    for r in REGIONS:
        total = sum(perf_weights[r].values())
        for m in perf_weights[r]:
            perf_weights[r][m] /= total

    # Create a thread-safe progress queue for UI updates
    progress_queue = Queue()

    with torch.no_grad():
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            ref_img = nib.load(reference_image_path)
            patient_id = extract_patient_id(reference_image_path)
            print(f"\nProcessing patient: {patient_id}\n")

            w_ttd, w_tta = 0.5, 0.5
            model_predictions = {}
            model_uncertainties = {}

            # Prepare arguments for concurrent inference for each model
            args_list = []
            futures_list = []
            for model_name, (model, inferer) in models_dict.items():
                args_list.append((model_name, model, inferer, image, config.device, n_iterations, w_ttd, w_tta))
            
            # Run model inference concurrently using ThreadPoolExecutor
            last_progress = [0]
            futures_list = []
            with ThreadPoolExecutor(max_workers=len(args_list)) as executor:
                for args in args_list:
                    future = executor.submit(compute_model_inference, args, progress_queue)
                    futures_list.append(future)

                # While the futures are running, poll the progress queue to update the progress bar.
                while not all(f.done() for f in futures_list):
                    update_progress_bar(progress_bar, progress_queue, last_progress)
                    time.sleep(0.1)

                # Flush any remaining progress updates.
                update_progress_bar(progress_bar, progress_queue, last_progress)
                
                # Retrieve results from all futures.
                for future in as_completed(futures_list):
                    mname, adjusted_prediction, combined_uncertainty = future.result()
                    model_predictions[mname] = adjusted_prediction
                    model_uncertainties[mname] = combined_uncertainty

                progress_bar.progress(100, text="100% completed")

            fused_logits = None
            for name, pred in model_predictions.items():
                w_log = torch.stack([
                    pred[i] * perf_weights[REGIONS[i]][name]
                    for i in range(len(REGIONS))
                ])
                fused_logits = w_log  if fused_logits is None else fused_logits +  w_log

            T_opt = 100*4.117968559265137
            probabilities = torch.softmax(fused_logits.float()/T_opt, dim=0)
            seg = probabilities.argmax(dim=0)

            # Fuse uncertainty maps for tumor regions
            fused_uncertainty = {}
            for idx, region in enumerate(["NCR", "ED", "ET"]):
                uncertainty_sum = torch.zeros_like(model_uncertainties[next(iter(model_uncertainties))][idx+1])
                for model_name in model_uncertainties:
                    weight = perf_weights[region][model_name]
                    uncertainty_sum += weight * model_uncertainties[model_name][idx+1]
                fused_uncertainty[region] = minmax_uncertainties(uncertainty_sum.cpu().numpy())

            # Save all outputs
            output_path = os.path.join(output_dir, f"softmax_{patient_id}.nii.gz")
            save_probability_map_as_nifti(probabilities, ref_img, output_path)

            output_path = os.path.join(output_dir, f"segmentation_{patient_id}.nii.gz")
            seg = seg.squeeze(0)
            save_segmentation_as_nifti(seg, ref_img, output_path)

            # Save uncertainty maps
            for region in ["NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)

            torch.cuda.empty_cache()

# if __name__ == "__main__":
#     patient_id = "01556"
#     models_dict = load_all_models()
#     composite_score_weights = {"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}
#     test_loader = dataloaders.load_test_data(config.json_path, config.root_dir)
#     ensemble_segmentation(test_loader, models_dict, composite_score_weights, n_iterations=1, patient_id=patient_id, output_dir="./assets/segmentations")
