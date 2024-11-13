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
import pandas as pd
import torch.nn as nn
from monai.losses import DiceLoss
from uncertainty.test_time_dropout import (
    test_time_dropout_inference,
    scale_to_range_0_100,
)
from mri_feature_extraction import extract_features_from_tensor
from sklearn.preprocessing import StandardScaler
from monai.metrics import DiceMetric
import optuna 

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
        "attentionunet": models.attunet_model,
    }
    
    loaded_models = {}
    for model_name, model_class in model_map.items():
        model = model_class.to(config.device).eval()
        checkpoint = torch.load(config.model_paths[model_name], map_location=config.device)
        model.load_state_dict(checkpoint["state_dict"])
        loaded_models[model_name] = model
    
    return loaded_models

class LinearEnsemblePredictor(nn.Module):
    def __init__(self, input_dim=79, hidden_dim1=64, hidden_dim2=32, output_dim=12, dropout_rate=0.3):
        super(LinearEnsemblePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
        # Xavier initialization for all linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(-1, 4, 3)  # Reshape for 4 models × 3 tissues
        x = torch.softmax(x, dim=1)  # Softmax per model across each tissue
        return x

models_dict = load_all_models()

# performance_metrics = pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/model_weights.csv", index_col=0).to_dict(orient="index")
performance_metrics = pd.DataFrame({
    'Model': ['attentionunet', 'segresnet', 'swinunetr', 'vnet'],
    '1': [0.5122, 0.7865, 0.6239, 0.5282],
    '2': [0.6987, 0.7233, 0.7149, 0.6758],
    '4': [0.5867, 0.6789, 0.7504, 0.7218]
})

# Instantiate Meta-Learner
# meta_learner = LinearEnsemblePredictor().to(config.device)
# optimizer = torch.optim.Adam(meta_learner.parameters(), lr=0.0001, weight_decay=1e-4)
dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

features_list = []

class_labels = [0, 1, 2, 4]  # 0 is background, 1: NCR, 2: ED, 4: ET
tissue_types = {1: "NCR", 2: "ED", 4: "ET"} 
num_classes = len(class_labels)
threshold = 0.5
feature_scaler = StandardScaler()  # For feature scaling

def objective(trial):
    # Sample hyperparameters
    hidden_dim1 = trial.suggest_int("hidden_dim1", 32, 128)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 16, 64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)

    meta_learner = LinearEnsemblePredictor(
        hidden_dim1=hidden_dim1, 
        hidden_dim2=hidden_dim2, 
        dropout_rate=dropout_rate
    ).to(config.device)
    optimizer = torch.optim.Adam(meta_learner.parameters(), lr=learning_rate, weight_decay=weight_decay)

    n_epochs = 2

    for epoch in range(n_epochs):
        meta_learner.train()
        total_loss = 0  # Initialize total loss for accumulation
        num_batches = len(config.val_loader_subset)

        for idx, batch_data in enumerate(config.val_loader_subset): 
            image = batch_data["image"].to(config.device)  
            ground_truth = batch_data["label"].to(config.device).float() 
            
            optimizer.zero_grad()

            # Prepare model predictions and features list
            model_predictions = []
            combined_features = []

            with torch.no_grad():  
                for model_name, model in models_dict.items():
                    # Get model prediction
                    model_inferer = partial(
                        sliding_window_inference,
                        roi_size=config.roi,
                        sw_batch_size=config.sw_batch_size,
                        predictor=model,
                        overlap=config.infer_overlap,
                    )
                    
                    prob = torch.sigmoid(model_inferer(image)).to(config.device)
                    seg = (prob[0].detach().cpu().numpy() > threshold).astype(np.int8)
                    seg_out = np.zeros(
                        (seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.int8
                    )
                    seg_out[seg[1] == 1] = 2
                    seg_out[seg[0] == 1] = 1
                    seg_out[seg[2] == 1] = 4
                    model_predictions.append(seg_out)

                    # Compute uncertainty for the model
                    mean_output, variance_output = test_time_dropout_inference(model, image, model_inferer)
                    if isinstance(variance_output, np.ndarray):
                        variance_output = scale_to_range_0_100(variance_output)
                        variance_output = torch.from_numpy(variance_output).to(config.device)
                    uncertainty_ncr = torch.mean(variance_output[:, 0]).item()
                    uncertainty_ed = torch.mean(variance_output[:, 1]).item()
                    uncertainty_et = torch.mean(variance_output[:, 2]).item()

                    # Collect MRI features and combine them with uncertainties
                    mri_features = extract_features_from_tensor(image[0])

                    # Build the feature vector by collecting all relevant features
                    feature_vector = []
                    for mod in ["flair", "t1", "t1ce", "t2"]:
                        feature_vector.extend([
                            mri_features[f'{mod}_mean_intensity'],
                            mri_features[f'{mod}_stddev_intensity'],
                            mri_features[f'{mod}_entropy'],
                            mri_features[f'{mod}_sobel_mean'],
                            mri_features[f'{mod}_sobel_std'],
                            mri_features[f'{mod}_gabor_mean_freq_0.1'],
                            mri_features[f'{mod}_gabor_std_freq_0.1'],
                            mri_features[f'{mod}_gabor_mean_freq_0.5'],
                            mri_features[f'{mod}_gabor_std_freq_0.5'],
                            mri_features[f'{mod}_gabor_mean_freq_0.8'],
                            mri_features[f'{mod}_gabor_std_freq_0.8'],
                            mri_features[f'{mod}_lbp_mean'],
                            mri_features[f'{mod}_lbp_std'],
                            mri_features[f'{mod}_skewness'],
                            mri_features[f'{mod}_kurtosis'],
                            mri_features[f'{mod}_glcm_contrast_mean'],
                            mri_features[f'{mod}_glcm_contrast_std'],
                            mri_features[f'{mod}_gradient_magnitude_mean']
                        ])

                    # Add cross-modality features
                    feature_vector.append(mri_features['combined_center_of_mass'][0])  # x-coordinate of center of mass
                    feature_vector.append(mri_features['combined_center_of_mass'][1])  # y-coordinate of center of mass
                    feature_vector.append(mri_features['combined_center_of_mass'][2])  # z-coordinate of center of mass
                    feature_vector.append(mri_features['combined_entropy'])  # combined entropy

                    # Add uncertainty metrics to the feature vector
                    feature_vector.extend([uncertainty_ncr, uncertainty_ed, uncertainty_et])

                    # Convert the feature vector to tensor format and add to the combined features list
                    combined_features.append(feature_vector)

            model_predictions = np.array(model_predictions)
            combined_features_np = np.array(combined_features)
            feature_scaler.partial_fit(combined_features_np)
            standardized_features = feature_scaler.transform(combined_features_np)
            combined_features_tensor = torch.tensor(standardized_features, dtype=torch.float32, device=config.device)

            # Predict weights
            predicted_weights = meta_learner(combined_features_tensor)
            
            # Initialize weighted votes as tensor on correct device
            weighted_votes = torch.zeros((len(class_labels),) + model_predictions[0].shape, device=config.device) 

            # Apply weighted voting for each tissue class (excluding background)
            for class_idx in class_labels[1:]:  # Skip background class (class_idx 0); [1, 2, 4] 
                class_vote_idx = class_labels.index(class_idx)
                for model_idx, model_seg in enumerate(model_predictions):
                    tissue_weight = predicted_weights[0, model_idx, class_vote_idx - 1].item()  # Get weight for tissue
                    weighted_votes[class_vote_idx] += (torch.tensor(model_seg == class_idx, device=config.device) * tissue_weight)

            # Determine tumor class with highest weighted vote per voxel
            final_segmentation_indices = torch.argmax(weighted_votes[1:], dim=0) + 1  

            # Apply background where votes are below threshold
            background_mask = (weighted_votes[1:].max(dim=0).values <= threshold)
            final_segmentation = torch.zeros_like(final_segmentation_indices, dtype=torch.int32)
            for i, class_label in enumerate(class_labels[1:]):
                final_segmentation[final_segmentation_indices == i + 1] = class_label
            final_segmentation[background_mask] = 0

            # Calculate Dice score and optimize
            loss = dice_loss(final_segmentation.unsqueeze(0), ground_truth)
            loss = loss.requires_grad_()  # Ensures it has grad_fn
            loss.backward()

            optimizer.step()

            # Accumulate loss for this batch
            total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Batch {idx + 1}, Loss: {loss.item():.4f}")

        # Average loss over all batches
        average_loss = total_loss / num_batches
        print(f"\tEpoch {epoch + 1} average loss: {average_loss:.4f}")
        torch.cuda.empty_cache() 
    return average_loss

    
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_params)
print("Best loss:", study.best_value)

print("Meta-learner training complete.")