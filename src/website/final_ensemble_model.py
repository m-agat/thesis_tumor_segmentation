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
        "vnet": load_model(models.vnet_model, config.model_paths["vnet"], config.device),
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
    """
    Save a probability map (float32) as a NIfTI file.
    """
    if isinstance(prob_map, torch.Tensor):
        prob_map = prob_map.cpu().numpy()
    # Make sure to keep the float values
    prob_map = prob_map.astype(np.float32)
    prob_img = nib.Nifti1Image(prob_map, affine=ref_img.affine, header=ref_img.header)
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
        hybrid_mean = np.squeeze((ttd_mean + tta_mean) / 2)
        hybrid_uncertainty = np.squeeze(w_ttd * ttd_uncertainty + w_tta * tta_uncertainty, axis=0)

    torch.cuda.synchronize()
    return model_name, hybrid_mean, hybrid_uncertainty

def update_progress_bar(progress_bar, progress_queue, last_progress):
    """Poll the progress queue and update the progress bar only if progress increases."""
    while not progress_queue.empty():
        pct = progress_queue.get()
        # Only update if the new percentage is greater than the last recorded value.
        if pct > last_progress[0]:
            last_progress[0] = pct
            progress_bar.progress(pct, text=f"{pct}% completed")

def ensemble_segmentation(test_loader, models_dict, composite_score_weights, n_iterations=10, patient_id=None, output_dir="./output_segmentations", progress_bar=None):
    os.makedirs(output_dir, exist_ok=True)
    test_data_loader = config.find_patient_by_id(patient_id, test_loader) if patient_id is not None else test_loader

    # Compute performance weights for all classes (BG, NCR, ED, ET)
    model_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
    for model_name in models_dict.keys():
        performance_weights_path = f"../models/performance/{model_name}/average_metrics.json"
        metrics = load_weights(performance_weights_path)
        composite_scores = compute_composite_scores(metrics, composite_score_weights)
        for region in ["BG", "NCR", "ED", "ET"]:
            model_weights[region][model_name] = composite_scores[region]
    for region in model_weights:
        total_weight = sum(model_weights[region].values())
        for model_name in model_weights[region]:
            model_weights[region][model_name] /= total_weight

    # Create a thread-safe progress queue for UI updates
    progress_queue = Queue()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_data in test_data_loader:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]
            ref_img = nib.load(reference_image_path)  # Load reference once per batch
            patient_id = extract_patient_id(reference_image_path)
            print(f"\nProcessing patient: {patient_id}\n")

            w_ttd, w_tta = 0.5, 0.5
            adjusted_weights = {region: {} for region in ["BG", "NCR", "ED", "ET"]}
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
                    mname, hybrid_mean, hybrid_uncertainty = future.result()
                    model_predictions[mname] = hybrid_mean
                    model_uncertainties[mname] = hybrid_uncertainty
                
                    for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                        alpha = 1
                        uncertainty_penalty = (1 - alpha * np.median(hybrid_uncertainty[idx]))
                        print("Uncertainty median ", np.median(hybrid_uncertainty[idx]))
                        print("Uncertainty penalty ", uncertainty_penalty)

                        adjusted_weights[region][mname] = model_weights[region][mname] * uncertainty_penalty
                        print(f"Adjusted weight for {mname}: ", adjusted_weights[region][mname])

                progress_bar.progress(100, text="100% completed")

            # Step 2: Normalize weights per region.
            for region in ["BG", "NCR", "ED", "ET"]:
                total_weight = sum(adjusted_weights[region].values())
                for model_name in adjusted_weights[region]:
                    adjusted_weights[region][model_name] /= total_weight
                    print(f"Model: {model_name}, Region: {region}, Final Weight: {adjusted_weights[region][model_name]:.3f}")

            # Step 3: Fuse predictions using the adjusted weights.
            weighted_logits = {region: [] for region in ["BG", "NCR", "ED", "ET"]}
            for model_name in models_dict.keys():
                # Convert the stored NumPy logits map to a torch tensor.
                # Expected shape: [num_classes, H, W, D]
                logits = torch.from_numpy(model_predictions[model_name]).to(config.device)
                for idx, region in enumerate(["BG", "NCR", "ED", "ET"]):
                    weight = torch.tensor(adjusted_weights[region][model_name], dtype=torch.float32, device=config.device)
                    region_logits = logits[idx]
                    weighted_logits[region].append(weight * region_logits)

            # Fuse logits by summing weighted contributions.
            fused_bg = torch.sum(torch.stack(weighted_logits["BG"]), dim=0)
            fused_tumor = [torch.sum(torch.stack(weighted_logits[region]), dim=0) for region in ["NCR", "ED", "ET"]]
            fused_logits = torch.stack([fused_bg] + fused_tumor, dim=0)

            # Convert the ensembled logits to probability maps by applying softmax.
            fused_probs = torch.softmax(fused_logits, dim=0)
            seg = fused_probs.argmax(dim=0).unsqueeze(0)


            # --- Fuse uncertainty maps for all classes (BG, NCR, ED, ET) ---
            fused_uncertainty = {}
            for idx, region in enumerate(["NCR", "ED", "ET"]):
                uncertainty_sum = torch.zeros_like(torch.from_numpy(model_uncertainties[next(iter(model_uncertainties))][idx+1]))
                for model_name in model_uncertainties:
                    weight = adjusted_weights[region][model_name]
                    uncertainty_sum += weight * torch.from_numpy(model_uncertainties[model_name][idx+1])
                fused_uncertainty[region] = minmax_uncertainties(uncertainty_sum.cpu().numpy())
            
            # --- Save probability maps ---
            output_path = os.path.join(
                output_dir, f"hybrid_softmax_{patient_id}.nii.gz"
            )
            save_probability_map_as_nifti(fused_probs, ref_img, output_path)

            # --- Save segmentation ---
            output_path = os.path.join(
                output_dir, f"hybrid_segmentation_{patient_id}.nii.gz"
            )
            seg = seg.squeeze(0)  # remove batch dimension
            save_segmentation_as_nifti(seg, ref_img, output_path)

            # --- Save uncertainty maps --- 
            for region in ["NCR", "ED", "ET"]:
                output_path = os.path.join(output_dir, f"uncertainty_{region}_{patient_id}.nii.gz")
                save_uncertainty_as_nifti(fused_uncertainty[region], ref_img, output_path)
            
            global_uncertainty = np.zeros_like(fused_uncertainty["NCR"]) # Use one of the maps as shape reference 
            seg_np = seg.cpu().numpy() # Convert segmentation to a numpy array
            region_to_label = {"NCR": 1, "ED": 2, "ET": 3}
            for region, label in region_to_label.items(): 
                region_mask = (seg_np == label) 
                global_uncertainty[region_mask] = fused_uncertainty[region][region_mask]

            global_output_path = os.path.join(output_dir, f"uncertainty_global_{patient_id}.nii.gz") 
            save_uncertainty_as_nifti(global_uncertainty, ref_img, global_output_path) 
            print(f"Global uncertainty map saved to {global_output_path}")

            torch.cuda.empty_cache()

# if __name__ == "__main__":
#     patient_id = "00657"
#     train_loader, _ = dataloaders.get_loaders(
#         batch_size=config.batch_size,
#         json_path=config.json_path,
#         basedir=config.root_dir,
#         fold=None,
#         roi=config.roi,
#         use_final_split=True,
#     )
#     models_dict = load_all_models()
#     composite_score_weights = {"Dice": 0.45, "HD95": 0.15, "Sensitivity": 0.3, "Specificity": 0.1}
#     ensemble_segmentation(train_loader, models_dict, composite_score_weights, n_iterations=10, patient_id=patient_id)
