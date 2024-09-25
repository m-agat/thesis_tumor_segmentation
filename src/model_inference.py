import torch
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
import os 
from src.metrics import compute_dice_score
from src.utils import resample_to_shape

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
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
    return seg_out

def get_segmentation_softmax(model, roi, test_loader, overlap=0.6, device="cuda"):
    """
    Perform segmentation using sliding window inference with softmax activation (multi-class segmentation).
    Returns soft predictions (class probabilities).
    Use for the weighted ensemble
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
            prob = torch.softmax(model_inferer(image), dim=1)  # Use softmax for multi-class
            seg = prob.detach().cpu().numpy()

    # Return the soft predictions (probabilities for each class)
    return seg

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
    Perform ensemble segmentation and compute uncertainty based on prediction variance
    and Dice score for multi-class segmentation.
    """
    # Initialize a list to store all model predictions
    all_preds = []

    # Loop through the models and get segmentation predictions, including nnUNet
    for model_name, model in models.items():
        if model_name == 'nnUNet':
            # nnUNet is a precomputed result, so we use it directly
            seg_out = model
        else:
            # Other models are Torch models and need to run inference
            seg_out = get_segmentation(model, roi, test_loader, overlap)

        all_preds.append(seg_out)

    # Convert predictions to a NumPy array of shape [num_models, H, W, D] (for each class)
    all_preds = np.array(all_preds)

    # Calculate mean prediction for each voxel for each class
    num_classes = 5
    ensemble_class_preds = np.zeros((num_classes,) + all_preds[0].shape)  # Shape: (num_classes, H, W, D)

    # For each class, calculate the mean prediction over all models for that class
    for class_idx in range(num_classes):
        # Calculate mean probability for each class
        ensemble_class_preds[class_idx] = np.mean(all_preds == class_idx, axis=0)

    # Get final segmentation by taking the argmax over the class dimension (i.e., the most confident class for each voxel)
    final_segmentation = np.argmax(ensemble_class_preds, axis=0)

    # Calculate variance-based uncertainty based on the variance of predictions for each class
    variance_uncertainty_map = np.var(all_preds, axis=0)

    # Calculate Dice score-based uncertainty
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in range(num_classes):
        # Get Dice uncertainty for each class
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    # Adjust variance-based uncertainty based on accuracy
    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    # Return the final segmentation along with both uncertainty maps
    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map

def get_weighted_ensemble_segmentation_with_uncertainty(models, test_loader, roi, ground_truth, model_dice_scores, overlap=0.6, num_classes=3):
    """
    Perform weighted ensemble segmentation with uncertainty based on model accuracy (Dice scores).
    Each model's prediction is weighted by its performance.
    """
    all_preds = []
    total_weight = 0

    # Define the target shape for resampling
    target_shape = None

    # Loop through the models and get segmentation predictions
    for model_name, model in models.items():
        weight = model_dice_scores.get(model_name, 1.0)  # Default weight if not found

        if model_name == 'nnUNet':
            # nnUNet is a precomputed segmentation result, reshape to (3, H, W, D)
            # Assuming that nnUNet already produces a segmentation map, split it into 3 binary maps for each class
            seg_out = np.zeros((num_classes,) + model.shape)  # Create empty array for 3 classes
            seg_out[0] = (model == 1).astype(np.int8)  # Class 1
            seg_out[1] = (model == 2).astype(np.int8)  # Class 2
            seg_out[2] = (model == 4).astype(np.int8)  # Class 3
        else:
            # Other models require inference with softmax
            seg_out = get_segmentation_softmax(model, roi, test_loader, overlap)

        # Log the shape of the output before resampling
        print(f"{model_name} output shape before resampling: {seg_out.shape}")

        # Set the target shape based on the first model (or the shape you want to resample to)
        if target_shape is None:
            target_shape = seg_out.shape[1:]  # Set the target shape to the shape of the first model's output

        # Resample the segmentation to the target shape if needed
        if seg_out.shape[1:] != target_shape:
            seg_out = np.array([resample_to_shape(seg, target_shape) for seg in seg_out])

        # Log the shape after resampling
        print(f"{model_name} output shape after resampling: {seg_out.shape}")

        # Apply model's weight (Dice score) to the segmentation
        weighted_seg = seg_out * weight
        all_preds.append(weighted_seg)
        total_weight += weight

    # Stack predictions: resulting shape is [num_models, num_classes, H, W, D]
    all_preds = np.stack(all_preds, axis=0)

    # Initialize ensemble predictions for each class
    ensemble_class_preds = np.zeros((num_classes,) + all_preds[0].shape[1:])  # Shape: (num_classes, H, W, D)

    # For each class, calculate the weighted average prediction over all models for that class
    for class_idx in range(num_classes):
        ensemble_class_preds[class_idx] = np.sum(all_preds[:, class_idx, :, :, :], axis=0) / total_weight

    # Get final segmentation by taking the argmax over the class dimension
    final_segmentation = np.argmax(ensemble_class_preds, axis=0)

    # Calculate variance-based uncertainty across the weighted predictions
    variance_uncertainty_map = np.var(all_preds, axis=0)

    # Calculate Dice score-based uncertainty
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in range(num_classes):
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    # Adjust variance-based uncertainty based on accuracy
    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    # Return the final segmentation along with both uncertainty maps
    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map




