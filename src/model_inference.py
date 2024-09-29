import torch
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
import os 
from src.metrics import compute_dice_score

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
            prob = torch.softmax(model_inferer(image), dim=1) 
            seg = prob.detach().cpu().numpy()

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
    all_preds = []

    for model_name, model in models.items():
        if model_name == 'nnUNet':
            seg_out = model
        else:
            seg_out = get_segmentation(model, roi, test_loader, overlap)

        all_preds.append(seg_out)

    # shape [num_models, H, W, D] (for each class)
    all_preds = np.array(all_preds)

    num_classes = 5
    ensemble_class_preds = np.zeros((num_classes,) + all_preds[0].shape)  # Shape: (num_classes, H, W, D)

    for class_idx in range(num_classes):
        # Calculate mean probability for each class
        ensemble_class_preds[class_idx] = np.mean(all_preds == class_idx, axis=0)

    # argmax over the class dimension (i.e., the most confident class for each voxel)
    final_segmentation = np.argmax(ensemble_class_preds, axis=0)

    # variance-based uncertainty based on the variance of predictions for each class
    variance_uncertainty_map = np.var(all_preds, axis=0)

    # Dice score-based uncertainty
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in range(num_classes):
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map


def get_weighted_ensemble_segmentation_with_uncertainty(models, test_loader, roi, ground_truth, overlap=0.6, weights=None):
    """
    Perform ensemble segmentation with model weighting based on Dice scores,
    and compute uncertainty based on prediction variance and Dice score for multi-class segmentation.
    """
    all_preds = []

    for model_name, model in models.items():
        if model_name == 'nnUNet':
            seg_out = model  # Use precomputed nnUNet segmentation
        else:
            seg_out = get_segmentation(model, roi, test_loader, overlap)

        all_preds.append(seg_out)

    # Convert predictions to numpy arrays for easier processing
    all_preds = np.array(all_preds)  # Shape: [num_models, H, W, D]

    num_classes = 5  # Example: You can adjust based on your number of classes
    ensemble_class_preds = np.zeros((num_classes,) + all_preds[0].shape)  # Shape: (num_classes, H, W, D)

    # Compute weighted mean for each class across models
    for class_idx in range(num_classes):
        weighted_sum = np.zeros_like(all_preds[0])

        # Sum the weighted predictions for each class across models
        for model_idx, pred in enumerate(all_preds):
            if weights:
                weighted_sum += (pred == class_idx) * weights[list(models.keys())[model_idx]]
            else:
                weighted_sum += (pred == class_idx)

        # Normalize by sum of weights or number of models (for the unweighted case)
        if weights:
            ensemble_class_preds[class_idx] = weighted_sum / np.sum(list(weights.values()))
        else:
            ensemble_class_preds[class_idx] = weighted_sum / len(all_preds)

    # Argmax over the class dimension (i.e., the most confident class for each voxel)
    final_segmentation = np.argmax(ensemble_class_preds, axis=0)

    # Variance-based uncertainty based on the variance of predictions for each class
    variance_uncertainty_map = np.var(all_preds, axis=0)

    # Dice score-based uncertainty
    dice_uncertainty_map = np.zeros_like(final_segmentation, dtype=np.float32)
    for class_idx in range(num_classes):
        dice_score = compute_dice_score(final_segmentation == class_idx, ground_truth == class_idx)
        dice_uncertainty_map[final_segmentation == class_idx] = 1 - dice_score

    # Compare to ground truth and calculate accuracy map
    accuracy_map = compare_predictions_to_ground_truth(final_segmentation, ground_truth)

    adjusted_variance_uncertainty = adjust_uncertainty_by_accuracy(variance_uncertainty_map, accuracy_map)

    return final_segmentation, adjusted_variance_uncertainty, dice_uncertainty_map







