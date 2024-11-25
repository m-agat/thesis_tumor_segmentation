import os
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import gc
import config.config as config
import models.models as models
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import nibabel as nib 

# Set base output directory
base_path = config.output_dir
os.makedirs(base_path, exist_ok=True)
print(f"Output directory: {base_path}")

# Load model function
def load_all_models(): 
    model_map = {
        "swinunetr": models.swinunetr_model,
        "segresnet": models.segresnet_model,
        "vnet": models.vnet_model,
        "attunet": models.attunet_model,
    }
    
    loaded_models = {}
    for model_name, model_class in model_map.items():
        model = model_class.to(config.device).eval()
        checkpoint = torch.load(config.model_paths[model_name], map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        loaded_models[model_name] = model
    
    return loaded_models

models_dict = load_all_models()
threshold = 0.5
for idx, batch_data in enumerate(config.val_loader): 
    image = batch_data["image"].to(config.device)  
    ground_truth = batch_data["label"][0].to(config.device)
    patient_path = batch_data["path"]

    print(f"Processing patient {idx + 1}/{len(config.val_loader)}: {patient_path[0]}")
    with torch.no_grad():  
        for model_name, model in models_dict.items():
            model_output_dir = os.path.join(base_path, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Get model prediction
            model_inferer = partial(
                sliding_window_inference,
                roi_size=config.roi,
                sw_batch_size=config.sw_batch_size,
                predictor=model,
                overlap=config.infer_overlap,
            )
            
            logits = model_inferer(image)
            # print(logits.shape)
            # prob = torch.sigmoid(logits).to(config.device)
            # seg = (prob[0].detach().cpu().numpy() > threshold).astype(np.int8)
            # seg_out = np.zeros(
            #     (seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8
            # )
            # seg_out[seg[1] == 1] = 2
            # seg_out[seg[0] == 1] = 1
            # seg_out[seg[2] == 1] = 4

            original_image_path = os.path.join(config.val_folder, patient_path[0], f"{patient_path[0]}_flair.nii.gz")
            original_nifti = nib.load(original_image_path)
            affine = original_nifti.affine
            # final_nifti = nib.Nifti1Image(seg_out, affine)

            # # Save the thresholded output
            # save_path = os.path.join(model_output_dir, f"{patient_path[0]}_{model_name}_segmentation.nii.gz")
            # nib.save(final_nifti, save_path)
            # print(f"Saved ensemble segmentation for patient {patient_path[0]} at {save_path}")

            # Save the logits maps as well
            # prob_out = prob.detach().cpu().numpy()
            logits_np = logits[0].cpu().numpy()
            prob_nifti = nib.Nifti1Image(logits_np, affine)
            prob_save_path = os.path.join(model_output_dir, f"{patient_path[0]}_{model_name}_logits.nii.gz")
            nib.save(prob_nifti, prob_save_path)
            print(f"Saved {model_name} logits for patient {patient_path[0]} at {prob_save_path}")
