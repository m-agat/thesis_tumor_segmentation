import numpy as np
import scipy.ndimage as ndimage
import torch

def postprocess_segmentation(pred):
    """
    Applies post-processing to the predicted segmentation to refine the result.
    Args:
        pred (torch.Tensor): The predicted segmentation mask.
    Returns:
        torch.Tensor: The post-processed segmentation mask.
    """
    pred_np = pred.cpu().numpy()
    post_processed_pred = np.zeros_like(pred_np)
    
    # Iterate over each class (e.g., WT, TC, ET)
    for class_idx in range(1, pred_np.max() + 1):
        class_mask = pred_np == class_idx
        
        # Remove small isolated components
        class_mask = remove_small_components(class_mask, min_size=1000)
        
        # Keep only the largest connected component
        class_mask = keep_largest_component(class_mask)
        
        # Apply morphological operations (optional)
        class_mask = ndimage.binary_closing(class_mask, structure=np.ones((3,3,3)))
        
        # Reconstruct final mask
        post_processed_pred[class_mask] = class_idx
    
    return torch.tensor(post_processed_pred).to(pred.device)

def remove_small_components(mask, min_size=1000):
    """Remove small connected components from a binary mask."""
    labeled, num_features = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    mask_sizes = sizes < min_size
    remove_pixel = mask_sizes[labeled]
    mask[remove_pixel] = 0
    return mask

def keep_largest_component(mask):
    """Keep only the largest connected component in a binary mask."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_component = (labeled == (np.argmax(sizes) + 1))
    return largest_component
