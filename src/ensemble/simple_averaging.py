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
#### Load models ####
#####################


def load_model(model_class, checkpoint_path, device):
    """
    Load the model from a checkpoint path.
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


def save_segmentation_as_nifti(
    predicted_segmentation, reference_image_path, output_path
):
    """
    Save the predicted segmentation as a NIfTI file.

    Parameters:
    - predicted_segmentation: The segmentation output as a tensor or numpy array.
    - reference_image_path: Path to the reference NIfTI image (to copy affine and header).
    - output_path: Path where the new segmentation NIfTI file will be saved.
    """
    # Convert to NumPy array if it's a tensor
    if isinstance(predicted_segmentation, torch.Tensor):
        predicted_segmentation = predicted_segmentation.cpu().numpy()

    # Ensure correct data type
    predicted_segmentation = predicted_segmentation.astype(np.uint8)  # or np.int16

    # Load reference image to get affine and header
    ref_img = nib.load(reference_image_path)

    # Create a new NIfTI image with the segmentation
    seg_img = nib.Nifti1Image(
        predicted_segmentation, affine=ref_img.affine, header=ref_img.header
    )

    # Save as .nii.gz file
    nib.save(seg_img, output_path)

    print(f"Segmentation saved to {output_path}")


# Load multiple models
models_dict = {
    "swinunetr": load_model(
        models.swinunetr_model, config.model_paths["swinunetr"], config.device
    ),
    "segresnet": load_model(
        models.segresnet_model, config.model_paths["segresnet"], config.device
    ),
    "attunet": load_model(
        models.attunet_model, config.model_paths["attunet"], config.device
    ),
    "vnet": load_model(models.vnet_model, config.model_paths["vnet"], config.device),
}

# Post-processing transforms
post_activation = Activations(softmax=True)  # Softmax for multi-class output
post_pred = AsDiscrete(argmax=True, to_onehot=4)  # Convert to one-hot encoding

output_dir = "./output_segmentations"
os.makedirs(output_dir, exist_ok=True)

########################################
#### Perform ensemble inference ####
########################################

patient_id = "01556"
one_patient = config.find_patient_by_id(patient_id, config.test_loader)

with torch.no_grad():
    for idx, batch_data in enumerate(one_patient):
        image = batch_data["image"].to(config.device)
        patient_path = batch_data["path"]
        ground_truth = batch_data["label"][0].cpu().numpy()

        # Collect logits from each model
        logits_list = []
        for model_name, (model, inferer) in models_dict.items():
            logits = inferer(image)
            print("logits shape: ", logits.shape)
            logits_list.append(logits)

        # # Average the logits across all models
        logits_tensor = torch.stack(
            logits_list, dim=0
        )  # Shape: (num_models, 4, 240, 240, 155)
        avg_logits = torch.mean(logits_tensor, dim=0)  # Shape: (4, 240, 240, 155)

        val_outputs_list = list(torch.unbind(avg_logits, dim=0))
        val_output_convert = [
            post_pred(post_activation(val_pred_tensor))
            for val_pred_tensor in val_outputs_list
        ]

        seg = val_output_convert[0]
        if hasattr(seg, "cpu"):
            seg = seg.cpu().numpy()

        if seg.ndim == 4:
            seg = seg.argmax(axis=0)

        output_path = os.path.join(output_dir, f"segmentation_{patient_id}.nii.gz")
        reference_image_path = patient_path[0]
        save_segmentation_as_nifti(seg, reference_image_path, output_path)

        print("Updated segmentation shape:", seg.shape)

        slice_index = seg.shape[-1] // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(seg[:, :, slice_index], cmap="gray")
        plt.title(f"Segmentation Slice at Index {slice_index}")
        plt.axis("off")
        plt.savefig("ensemble_map.png")
