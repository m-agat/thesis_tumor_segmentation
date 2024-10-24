import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

def get_augmentations():
    """
    Define the augmentations to be used for TTA.
    You can customize this list based on the type of variations you want to handle.
    """
    return [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
        transforms.ElasticTransform(alpha=30.0, sigma=2.0,),  # Elastic translation
    ]


def apply_augmentation(image, augmentations):
    """
    Randomly selects and applies an augmentation from the list.
    
    Args:
        image: Input image tensor.
        augmentations: List of augmentation functions.
        
    Returns:
        Augmented image tensor.
    """
    aug = np.random.choice(augmentations)
    return aug(image)


def test_time_augmentation(model, input_data, device, n_iterations=10):
    """
    Run test-time augmentation to estimate uncertainty.
    
    Args:
        model: Trained model.
        input_data: Input image tensor (e.g., MRI scan).
        device: Device to run the model on (CPU or GPU).
        n_iterations: Number of times to run inference with augmentations.
        
    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    model.eval()  # Switch to evaluation mode
    augmentations = get_augmentations()  # Prepare list of augmentations

    # Collect predictions over multiple augmented passes
    augmented_outputs = []
    with torch.no_grad():
        for _ in tqdm(range(n_iterations), desc="Predicting with TTA.."):
            # Apply a random augmentation to the input
            augmented_input = apply_augmentation(input_data, augmentations).to(device)
            
            # Forward pass through the model
            output = model(augmented_input)
            augmented_outputs.append(output.cpu().numpy())

    # Convert to NumPy array for easier manipulation
    augmented_outputs = np.array(augmented_outputs)
    
    # Compute mean and variance across iterations
    mean_output = np.mean(augmented_outputs, axis=0)
    variance_output = np.var(augmented_outputs, axis=0)
    
    return mean_output, variance_output
