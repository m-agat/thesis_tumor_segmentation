import os
import sys
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from functools import partial
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations

# Add custom modules
sys.path.append("../")
import models.models as models
import config.config as config

#####################
#### Load Models ####
#####################

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

#############################
#### Save Segmentation ######
#############################

def save_segmentation_as_nifti(predicted_segmentation, reference_image_path, output_path):
    """
    Save the predicted segmentation as a NIfTI file.

    Parameters:
    - predicted_segmentation: The segmentation output as a tensor or numpy array.
    - reference_image_path: Path to the reference NIfTI image (for affine and header copying).
    - output_path: Path where the new segmentation NIfTI file will be saved.
    """
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()

    predicted_segmentation = predicted_segmentation.astype(np.uint8)

    ref_img = nib.load(reference_image_path)
    seg_img = nib.Nifti1Image(predicted_segmentation, affine=ref_img.affine, header=ref_img.header)
    nib.save(seg_img, output_path)

    print(f"Segmentation saved to {output_path}")

########################################
#### Perform Ensemble Segmentation ####
########################################

def ensemble_segmentation(patient_id, test_loader, models_dict, output_dir="./output_segmentations"):
    """
    Perform segmentation using an ensemble of multiple models with simple averaging.

    Parameters:
    - patient_id: ID of the patient whose scan is being segmented.
    - test_loader: Dataloader for the test set.
    - models_dict: Dictionary containing trained models and their inferers.
    - output_dir: Directory where segmentations will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    one_patient = config.find_patient_by_id(patient_id, test_loader)

    post_activation = Activations(softmax=True)  # Convert logits to probabilities
    post_pred = AsDiscrete(argmax=True)  # Get the most probable class per voxel

    with torch.no_grad():
        for batch_data in one_patient:
            image = batch_data["image"].to(config.device)
            reference_image_path = batch_data["path"][0]  # Assuming first path is the reference

            # Collect logits from each model
            logits_list = []
            for model_name, (model, inferer) in models_dict.items():
                logits = inferer(image).squeeze(0)  # Remove batch dim -> (num_classes, H, W, D)
                logits_list.append(logits)

            # Average the logits across all models
            avg_logits = torch.mean(torch.stack(logits_list), dim=0)  # Shape: (num_classes, H, W, D)

            # Apply softmax and convert to segmentation map
            seg = torch.nn.functional.softmax(avg_logits, dim=0).argmax(dim=0).cpu().numpy()  # Shape: (H, W, D)

            # Save segmentation
            output_path = os.path.join(output_dir, f"segmentation_{patient_id}.nii.gz")
            save_segmentation_as_nifti(seg, reference_image_path, output_path)

            # Display a middle slice
            visualize_segmentation(seg, patient_id)

####################################
#### Visualize Segmentation ####
####################################

def visualize_segmentation(segmentation, patient_id):
    """
    Display and save a middle slice from the segmentation.

    Parameters:
    - segmentation: 3D NumPy array containing the segmentation result.
    - patient_id: ID of the patient (used for file naming).
    """
    slice_index = segmentation.shape[-1] // 2  # Middle slice

    plt.figure(figsize=(6, 6))
    plt.imshow(segmentation[:, :, slice_index], cmap="gray")
    plt.title(f"Segmentation Slice at Index {slice_index}")
    plt.axis("off")
    plt.savefig(f"segmentation_{patient_id}_slice.png")
    plt.show()

#######################
#### Run Inference ####
#######################

if __name__ == "__main__":
    patient_id = "01556"
    models_dict = load_all_models()
    ensemble_segmentation(patient_id, config.test_loader, models_dict)
