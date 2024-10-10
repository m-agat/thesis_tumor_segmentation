import torch
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
import os 
from src.metrics import compute_dice_score
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def load_models(device, results_dir):
    """
    Load pretrained models from the results directory.
    """
    from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet
    
    roi = (96, 96, 96)
    
    swinunetr_model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )

    segresnet_model = SegResNet(
        blocks_down=[1, 2, 2, 4],   
        blocks_up=[1, 1, 1],       
        init_filters=16,        
        in_channels=4,         
        out_channels=3,          
        dropout_prob=0.2            
    )

    attunet_model = AttentionUnet(
        spatial_dims=3,  
        in_channels=4,   
        out_channels=3,  
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),             
        dropout=0.0                       
    )

    vnet_model = VNet(
        spatial_dims=3, 
        in_channels=4,   
        out_channels=3, 
        act=("elu", {"inplace": True})
    )
        
    models = {
        'swinunetr_model.pt': swinunetr_model,
        'segresnet_model.pt': segresnet_model,
        'attunet_model.pt': attunet_model,
        'vnet_model.pt': vnet_model
    }

    for model_name, model in models.items():
        model.load_state_dict(torch.load(os.path.join(results_dir, model_name))["state_dict"])
        model.to(device)
        model.eval()
    
    return models

def get_segmentation(model, roi, test_loader, overlap=0.6, device="cuda"):
    """
    Perform segmentation using sliding window inference.
    """
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=overlap,
    )

    with torch.no_grad():
        for batch_data in test_loader:
            image = batch_data["image"].to(device)
            prob = torch.sigmoid(model_inferer(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2 # ED
            seg_out[seg[0] == 1] = 1 # NCR
            seg_out[seg[2] == 1] = 4 # ET
    return seg_out

def compare_predictions_to_ground_truth(prediction, ground_truth):
    """
    Compare the segmentation prediction with the ground truth.
    Returns a binary accuracy map (1 for correct, 0 for incorrect).
    """
    return (prediction == ground_truth).astype(np.int8)

def adjust_uncertainty_by_accuracy(uncertainty_map, accuracy_map):
    """
    Adjust the uncertainty map based on the accuracy of the prediction.
    High uncertainty in regions where the prediction was incorrect.
    """
    return uncertainty_map * (1 - accuracy_map)

def get_ensemble_segmentation_with_uncertainty(models, test_loader, roi, ground_truth, overlap=0.6):
    """
    Perform ensemble segmentation using majority voting and compute uncertainty 
    based on prediction variance and Dice score for multi-class segmentation.
    """
    all_preds = []

    for model_name, model in models.items():
        if model_name == 'nnUNet':
            seg_out = model  # Use precomputed nnUNet segmentation
        else:
            seg_out = get_segmentation(model, roi, test_loader, overlap)
        
        all_preds.append(seg_out)

    # shape: [num_models, H, W, D]
    all_preds = np.array(all_preds, dtype=np.int32)

    # Majority voting --> final segmentation
    # This will select the most common label across all models for each voxel
    final_segmentation = stats.mode(all_preds, axis=0, keepdims=True).mode[0]  # Using the mode for majority voting

    # Variance-based uncertainty: how much disagreement there is between models
    variance_uncertainty_map = np.var(all_preds, axis=0)

    # Dice score-based uncertainty
    num_classes = 5  # Including background, NCR, ED, ET
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in [0, 1, 2, 4]:  # Iterate over relevant classes
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    # Adjust variance uncertainty by accuracy map
    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map


def get_weighted_ensemble_segmentation_with_uncertainty(models, test_loader, roi, ground_truth, overlap=0.6, weights=None, class_size_factor=0.5):
    """
    Perform ensemble segmentation with model weighting based on Dice scores and class size,
    using weighted majority voting to combine model predictions.
    """
    print(f"The weights are: {weights}")
    all_preds = []

    tissue_types = {0: 'Background', 1: 'NCR', 2: 'ED', 4: 'ET'}

    # Get predictions from all models
    for model_name, model in models.items():
        if model_name == 'nnUNet':
            seg_out = model  # nnUNet segmentation is precomputed and used separately
        else:
            seg_out = get_segmentation(model, roi, test_loader, overlap)
        all_preds.append(seg_out)

    # All predictions from each model are gathered in one variable
    # Shape: [num_models, H, W, D]
    all_preds = np.array(all_preds)  

    # Use num_classes = [0, 1, 2, 4], but map these to indices 0-3
    class_labels = [0, 1, 2, 4]  # Relevant classes: background, NCR, ED, ET
    num_classes = len(class_labels)

    # Initialize an array to store weighted votes for all classes
    weighted_votes = np.zeros((num_classes,) + all_preds[0].shape, dtype=np.float32)  # Shape: (num_classes, H, W, D)

    # Compute class size weights to adjust the importance of each class
    # This way we try to account for the class imbalance (e.g. background is significantly bigger than the tumor)
    # So that the bigger classes do not overtake the predictions 
    class_sizes = {class_idx: np.sum(all_preds == class_idx) for class_idx in class_labels}
    total_voxels = np.prod(all_preds[0].shape)  # Total number of voxels in one prediction
    class_size_weights = {class_idx: (total_voxels / class_sizes[class_idx] if class_sizes[class_idx] > 0 else 0)
                          for class_idx in class_labels}
    for class_idx in class_size_weights:
        class_size_weights[class_idx] = (1 - class_size_factor) + class_size_factor * class_size_weights[class_idx]
    print(f"Class size weights: {class_size_weights}")

    # The predictions are combined using **weighted majority voting**
    # The predictions from each model are weighted by:
    # 1. Dice score per model and tissue 
    # 2. Class size to reduce impact of largeer classes
    for class_idx in class_labels:
        tissue_type = tissue_types.get(class_idx, None)

        # Map class labels [0, 1, 2, 4] to indices 0, 1, 2, 3 in the weighted_votes array
        class_vote_idx = class_labels.index(class_idx)

        for model_idx, pred in enumerate(all_preds):
            model_name = list(models.keys())[model_idx]

            if tissue_type in weights:
                if model_name in weights[tissue_type]:
                    # Apply model and class size weighting
                    weighted_votes[class_vote_idx] += (pred == class_idx) * weights[tissue_type][model_name] * class_size_weights[class_idx]
                else:
                    print(f"Model name '{model_name}' not found in weights for tissue {tissue_type}.")
                    raise KeyError(f"Model name '{model_name}' not found in weights for tissue {tissue_type}.")
            else:
                print(f"Missing weights for tissue type: {tissue_type}.")
                raise KeyError(f"Missing weights for tissue type: {tissue_type}.")

    # THE CLASS WITH THE HIGHEST WEIGHTED VOTE IS SELECTED FOR EACH VOXEL 
    # This will give us the index in class_labels (0 to 3), so we need to map it back to the original labels (0, 1, 2, 4)
    final_segmentation_indices = np.argmax(weighted_votes, axis=0)

    # Map the indices back to the original class labels (0, 1, 2, 4)
    final_segmentation = np.zeros_like(final_segmentation_indices, dtype=np.int32)
    for i, class_label in enumerate(class_labels):
        final_segmentation[final_segmentation_indices == i] = class_label

    # Uncertainty based on consensus
    # The higher the disagreement between the models' predictions, the higher the uncertainty
    consensus_agreement = np.mean(np.equal(all_preds, final_segmentation), axis=0)
    variance_uncertainty_map = 1 - consensus_agreement  

    # Uncertainty based on Dice score
    # Lower Dice scores --> higher uncertainty.
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in class_labels:
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    # Adjust variance uncertainty by accuracy map
    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map