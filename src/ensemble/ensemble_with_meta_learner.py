import os
import sys
sys.path.append("../")
import torch
import gc
import config.config as config
import models.models as models
from monai.inferers import sliding_window_inference
from functools import partial
import numpy as np
import pandas as pd
import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
from uncertainty.test_time_dropout import (
    test_time_dropout_inference,
    scale_to_range_0_100,
)

# Set base output directory
base_path = config.output_dir
os.makedirs(base_path, exist_ok=True)
print(f"Output directory: {base_path}")

# Load model function
def load_all_models():
    model_paths = {
    "swinunetr": "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/swinunetr_model.pt",
    "segresnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/segresnet_model.pt",
    "attentionunet": "/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/attunet_model.pt",
    "vnet": "/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/vnet_model.pt"
    }   
    model_map = {
        "swinunetr": models.swinunetr_model,
        "segresnet": models.segresnet_model,
        "vnet": models.vnet_model,
        "attentionunet": models.attunet_model,
    }
    
    loaded_models = {}
    for model_name, model_class in model_map.items():
        model = model_class.to(config.device).eval()
        checkpoint = torch.load(model_paths[model_name], map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        loaded_models[model_name] = model
    
    return loaded_models

models_dict = load_all_models()

# performance_metrics = pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/model_weights.csv", index_col=0).to_dict(orient="index")
performance_metrics = pd.DataFrame({
    'Model': ['attentionunet', 'segresnet', 'swinunetr', 'vnet'],
    '1': [0.6757, 0.6968, 0.7174, 0.6333],
    '2': [0.8245, 0.8378, 0.8443, 0.8093],
    '4': [0.8376, 0.8481, 0.8426, 0.8290]
})

# Meta-Learner Model Definition
class MetaLearner(nn.Module):
    def __init__(self, input_dim=14):
        super(MetaLearner, self).__init__()
        
        # Expanded architecture with residual connections
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 12)  # Output 12 values: 4 models × 3 tissues
        
        self.residual1 = nn.Linear(input_dim, 128)  # For residual connection
        self.residual2 = nn.Linear(128, 64)
        
        # Use layer normalization
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(64)
        self.ln3 = nn.LayerNorm(32)
        
    def forward(self, x):
        residual = self.residual1(x)
        x = torch.relu(self.ln1(self.fc1(x) + residual))  # Add residual connection after first layer
        
        residual = self.residual2(x)
        x = torch.relu(self.ln2(self.fc2(x) + residual))  # Add residual connection after second layer
        
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        
        # Reshape for model and tissue weights, with softmax per model dimension
        x = x.view(-1, 4, 3)
        x = torch.softmax(x, dim=1)  # Softmax per model across each tissue
        return x

# Instantiate Meta-Learner
meta_learner = MetaLearner().to(config.device)
optimizer = torch.optim.AdamW(meta_learner.parameters(), lr=0.0001)
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
focal_loss = FocalLoss(to_onehot_y=False)

features_list = []

n_epochs = 10
for epoch in range(n_epochs):
    meta_learner.train()
    epoch_loss = 0

    for idx, batch_data in enumerate(config.train_loader_subset):  # Assuming a DataLoader for training
        image = batch_data["image"].to(config.device)  # MRI scan tensor
        ground_truth = batch_data["label"].to(config.device).float()  # Ground truth segmentation

        # Inference for each model and collect predictions
        model_predictions = []
        features_list = []

        with torch.no_grad():  # Inference and feature extraction should be done without tracking gradients
            for model_name, model in models_dict.items():
                model_inferer = partial(
                    sliding_window_inference,
                    roi_size=config.roi,
                    sw_batch_size=config.sw_batch_size,
                    predictor=model,
                    overlap=config.infer_overlap,
                )
                
                prob = torch.sigmoid(model_inferer(image)).cpu()  # Probability map for each model
                model_predictions.append(prob)  # Shape: (B, 3, H, W, D) for each model
                
                # Extracting features for the meta-learner
                # Obtaining the uncertainty and summarizing it as a scalar
                mean_output, variance_output = test_time_dropout_inference(model, image, model_inferer)
                if isinstance(variance_output, np.ndarray):
                    variance_output = torch.from_numpy(variance_output).to(config.device)
                uncertainty_ncr, uncertainty_ed, uncertainty_et = variance_output[:, 0], variance_output[:, 1], variance_output[:, 2]
                uncertainty_ncr = torch.mean(uncertainty_ncr).item()
                uncertainty_ed = torch.mean(uncertainty_ed).item()
                uncertainty_et = torch.mean(uncertainty_et).item()

                # Obtain the performance metric for each model and tissue 
                perf_metrics = performance_metrics[performance_metrics["Model"] == model_name]
                dice_ncr, dice_ed, dice_et = perf_metrics[["1", "2", "4"]].values[0]

                # Get the brain scan statistics
                intensity_stats = []
                for modality in range(image.shape[1]):
                    modality_data = image[0, modality].detach().cpu().numpy()
                    intensity_stats.extend([np.mean(modality_data), np.std(modality_data)])

                features = [dice_ncr, dice_ed, dice_et, uncertainty_ncr, uncertainty_ed, uncertainty_et] + intensity_stats
                features_list.append(features)

        features_array = np.array(features_list, dtype=np.float32)  # Convert to NumPy array first
        features_tensor = torch.from_numpy(features_array).to(config.device)
        features_tensor = (features_tensor - features_tensor.mean(dim=0)) / (features_tensor.std(dim=0) + 1e-5)
        weights = meta_learner(features_tensor)  # Shape: (batch_size, 4 models, 3 tissues)

        prob_maps = {1: [], 2: [], 4: []}

        # Generate ensemble prediction using weighted probabilities
        for model_idx, model_pred in enumerate(model_predictions):
            for tissue_idx, tissue_label in enumerate([1, 2, 4]):  # NCR, ED, ET
                # Expand weights to match the shape of model_pred[:, tissue_idx]
                weight_expanded = weights[:, model_idx, tissue_idx].view(-1, 1, 1, 1)  # Adds spatial singleton dimensions
                weighted_prob = weight_expanded * model_pred[:, tissue_idx]
                prob_maps[tissue_label].append(weighted_prob.detach().cpu().numpy())  # Accumulate per model

        # Calculate final ensemble segmentation by averaging across the model dimension
        ensemble_prob = torch.zeros((1, 3, 96, 96, 96), dtype=torch.float32, requires_grad=True).to(config.device)

        for tissue_idx, tissue_label in enumerate([1, 2, 4]):  # NCR, ED, ET
            weighted_avg = np.mean(prob_maps[tissue_label], axis=0)  # Average across models
            # Remove the extra dimension (which should be of size 1)
            if weighted_avg.shape != (96, 96, 96):
                weighted_avg = np.mean(weighted_avg, axis=0)
            ensemble_prob.data[0, tissue_idx] = torch.tensor(weighted_avg, dtype=torch.float32).to(config.device)
        
        # Compute Dice loss between ensemble prediction and ground truth
        optimizer.zero_grad()
        loss = dice_loss(ensemble_prob, ground_truth)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} average loss: {epoch_loss / len(config.train_loader_subset):.4f}")

print("Meta-learner training complete.")