import os
import sys
import torch
import numpy as np
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
import re

sys.path.append("../")
import models.models as models
import config.config as config


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
            reference_image_path = batch_data["path"][0]
            patient_id = extract_patient_id(reference_image_path)
            model, inferer = next(iter(models_dict.values()))
            print(f"\nProcessing patient: {patient_id} using {model_name}\n")

            logits = inferer(image).squeeze(0)
            seg = torch.nn.functional.softmax(logits, dim=0).argmax(dim=0).unsqueeze(0)
            seg = seg.squeeze(0)

            output_path = os.path.join(output_dir, f"{model_name}_{patient_id}_pred_seg.nii.gz")
            save_segmentation_as_nifti(seg, reference_image_path, output_path)
            torch.cuda.empty_cache()
            # Process only one patient for this example.
            break


if __name__ == "__main__":
    # Change this variable to the model you want to use:
    selected_model = "attunet"  # Options: "attunet", "swinunetr", "segresnet"

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
    patient_id = "01529"

    ensemble_segmentation(config.test_loader, models_dict, patient_id=patient_id, output_dir=output_dir, model_name=selected_model)
