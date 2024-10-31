import torch
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    Rand3DElasticd,
)
import matplotlib.pyplot as plt
import os 

def get_augmentations():
    """
    Define the augmentations to be used for TTA.
    """
    return Compose([
        RandAdjustContrastd(keys=["image"], prob=1, gamma=(0.7, 1.5)),
        Rand3DElasticd(
            keys=["image"], 
            sigma_range=(2, 5),  
            magnitude_range=(100, 200), 
            prob=1.0, 
            rotate_range=(0.1, 0.1, 0.1), 
            shear_range=(0.1, 0.1, 0.1), 
            translate_range=(10, 10, 10)
        ),
    ])

def apply_augmentation(image, augmentations, visualize=False):
    """
    Randomly selects and applies an augmentation from the list, slice-by-slice for 3D images.
    
    Args:
        image: Input image tensor (B, C, D, H, W).
        augmentations: List of augmentation functions.
        visualize: Boolean flag to indicate whether to visualize original and augmented slices.
        
    Returns:
        Augmented image tensor.
    """
    data = {"image": image[0]}  # Use the first batch element
    augmented_data = augmentations(data)
    
    # Extract the augmented image
    augmented_image = augmented_data["image"].unsqueeze(0)  # Add batch dimension back
    
    if visualize:
        # Visualize a random slice
        slice_idx = image.shape[2] // 2  # Choose the middle slice for visualization
        original_slice = image[0, 0, slice_idx].cpu().numpy()
        augmented_slice = augmented_image[0, 0, slice_idx].cpu().numpy()
        visualize_augmentation(original_slice, augmented_slice, slice_idx)
    
    return augmented_image

def visualize_augmentation(original_slice, augmented_slice, slice_index, save_path="./augmentation_visualizations"):
    """
    Visualize the original and augmented slices side by side.
    
    Args:
        original_slice: Original 2D slice (H, W).
        augmented_slice: Augmented 2D slice (H, W).
        slice_index: Index of the slice being visualized.
    """
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f'Original Slice {slice_index}')
    axes[0].axis('off')
    
    axes[1].imshow(augmented_slice, cmap='gray')
    axes[1].set_title(f'Augmented Slice {slice_index}')
    axes[1].axis('off')
    
    plt.savefig(os.path.join(save_path, f"augmentation_slice_{slice_index}.png"))
    plt.close(fig)

def test_time_augmentation_inference(model_inferer, input_data, device, n_iterations=10):
    """
    Run test-time augmentation to estimate uncertainty.
    
    Args:
        model_inferer: Inference function (e.g., sliding window inferer).
        input_data: Input image tensor (e.g., MRI scan).
        device: Device to run the model on (CPU or GPU).
        n_iterations: Number of times to run inference with augmentations.
        
    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    augmentations = get_augmentations()  # Prepare list of augmentations

    # Collect predictions over multiple augmented passes
    augmented_outputs = []
    with torch.no_grad():
        for _ in tqdm(range(n_iterations), desc="Predicting with TTA.."):
            # Apply a random augmentation to the input
            augmented_input = apply_augmentation(input_data, augmentations, visualize=True).to(device)
            
            # Forward pass through the model using the inferer
            output = model_inferer(augmented_input)
            augmented_outputs.append(output.cpu().numpy())

    # Convert to NumPy array for easier manipulation
    augmented_outputs = np.array(augmented_outputs)
    
    # Compute mean and variance across iterations
    mean_output = np.mean(augmented_outputs, axis=0)
    variance_output = np.var(augmented_outputs, axis=0)
    
    return mean_output, variance_output