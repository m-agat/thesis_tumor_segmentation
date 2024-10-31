import sys
sys.path.append("../")
import torch
from functools import partial
from monai.inferers import sliding_window_inference
import config.config as config
import models.models as models
import pandas as pd 

model_paths = {
    "swinunetr": "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/swinunetr_model.pt",
    "segresnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/segresnet_model.pt",
    "attunet": "/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/attunet_model.pt",
    "vnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/vnet_model.pt"
}

class_weights = pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/model_weights.csv", index_col=0).to_dict(orient="index")

def load_model(model_class, checkpoint_path, device):
    # Instantiate the model
    model = model_class
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Send the model to the appropriate device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Set up the inference configuration using sliding window
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[config.roi[0], config.roi[1], config.roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6
    )
    
    return model, model_inferer

def weighted_class_average(ensemble_outputs, class_weights):
    combined_prob_map = torch.zeros_like(ensemble_outputs[0])
    
    class_indices = {"NCR": 1, "ED": 2, "ET": 4}
    
    for tissue_name, class_index in class_indices.items():
        # Initialize a weighted sum for the specific class
        weighted_sum = torch.zeros_like(ensemble_outputs[0][:, class_index])
        total_weight = 0
        
        # Apply class-specific weights for each model
        for model_name, output in ensemble_outputs.items():
            weight = class_weights[model_name][tissue_name]
            weighted_sum += weight * output[:, class_index]
            total_weight += weight
        
        # Normalize by the total weight
        combined_prob_map[:, class_index] = weighted_sum / total_weight
    
    return combined_prob_map


swinunetr, swinunetr_inferer = load_model(models.swinunetr_model, model_paths["swinunetr"], config.device)
segresnet, segresnet_inferer = load_model(models.segresnet_model, model_paths["segresnet"], config.device)
attunet, attunet_inferer = load_model(models.attunet_model, model_paths["attunet"], config.device)
vnet, vnet_inferer = load_model(models.vnet_model, model_paths["vnet"], config.device)

with torch.no_grad():
    for batch_data in config.test_loader:
        input_volume = batch_data["image"].cuda()
        patient_path = batch_data["path"]

        swinunetr_output = torch.softmax(swinunetr_inferer(input_volume))
        segresnet_output = torch.softmax(segresnet_inferer(input_volume))
        attunet_output = torch.softmax(attunet_inferer(input_volume))
        vnet_output = torch.softmax(vnet_inferer(input_volume))

        ensemble_outputs = {
            "swinunetr": swinunetr_output,
            "segresnet": segresnet_output,
            "attunet": attunet_output,
            "vnet": vnet_output
        }

        final_ensemble_output = weighted_class_average(ensemble_outputs, class_weights)

        final_prediction = torch.argmax(final_ensemble_output, dim=1)