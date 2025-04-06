from monai import transforms
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class ReversibleScaleIntensityd:
    def __init__(self, keys, factor):
        self.keys = keys
        self.factor = factor

    def __call__(self, data):
        # Multiply by (1 + factor)
        data["image"] = data["image"] * (1 + self.factor)
        return data


class InverseReversibleScaleIntensityd:
    def __init__(self, keys, factor):
        self.keys = keys
        self.factor = factor

    def __call__(self, data):
        # Divide by (1 + factor) to reverse the scaling
        data["image"] = data["image"] / (1 + self.factor)
        return data


def get_augmentations():
    """
    Define reversible augmentations for TTA.
    """
    augmentations = [
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=0),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=0),
        },
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=1),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=1),
        },
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=2),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=2),
        },
        {
            "aug": ReversibleScaleIntensityd(keys=["image"], factor=0.1),
            "inverse": InverseReversibleScaleIntensityd(keys=["image"], factor=0.1),
        },
        {
            "aug": transforms.ShiftIntensityd(keys="image", offset=0.1),
            "inverse": transforms.ShiftIntensityd(keys="image", offset=-0.1),
        },
        {
            "aug": transforms.RotateD(keys=["image"], angle=10, keep_size=True),
            "inverse": transforms.RotateD(keys=["image"], angle=-10, keep_size=True),
        },
        {
            "aug": transforms.RotateD(keys=["image"], angle=-10, keep_size=True),
            "inverse": transforms.RotateD(keys=["image"], angle=10, keep_size=True),
        },
    ]
    return augmentations


def apply_augmentation(image, aug, visualize=False):
    """
    Apply a single augmentation from the list.
    Args:
        image: Input image tensor (B, C, D, H, W).
        aug: The augmentation function to apply.
        visualize: Whether to visualize the original and augmented slices.
    Returns:
        Augmented image tensor.
    """
    data = {"image": image[0]}  # Use the first batch element
    augmented_data = aug(data)
    augmented_image = augmented_data["image"].unsqueeze(0)  # Add batch dimension back

    if visualize:
        slice_idx = image.shape[2] // 2  # Choose middle slice
        original_slice = image[0, 0, slice_idx].cpu().numpy()
        augmented_slice = augmented_image[0, 0, slice_idx].cpu().numpy()
        visualize_augmentation(original_slice, augmented_slice, slice_idx)

    return augmented_image


def reverse_augmentation(output, inverse_aug):
    """
    Reverse the augmentation applied to the output.
    Args:
        output: Predicted segmentation or output from the model.
        inverse_aug: Inverse augmentation function.
    Returns:
        Reversed output.
    """
    data = {"image": torch.tensor(output[0])}  # First batch element
    reversed_data = inverse_aug(data)
    reversed_output = reversed_data["image"].unsqueeze(0).numpy()  # Add batch dimension
    return reversed_output


def tta_variance(model_inferer, input_data, device, n_iterations=10, on_step=None):
    """
    Run test-time augmentation with reversible transformations for uncertainty estimation.
    Args:
        model_inferer: Inference function (e.g., sliding window inference).
        input_data: Input image tensor (e.g., MRI scan).
        device: Device to run the model on (CPU or GPU).
        n_iterations: Number of times to run inference with augmentations.
    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    augmentations = get_augmentations()  # Get list of augmentations with inverses
    augmented_outputs = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in tqdm(range(n_iterations), desc="Predicting with TTA.."):
            # update progress
            if on_step:
                on_step(i+1, n_iterations)

            # Randomly pick an augmentation
            aug_entry = np.random.choice(augmentations)
            aug = aug_entry["aug"]
            inverse_aug = aug_entry["inverse"]

            # Apply augmentation
            augmented_input = apply_augmentation(input_data, aug).to(device)

            # Forward pass
            pred = model_inferer(augmented_input)
            output = pred.cpu().numpy()

            # Reverse augmentation
            reversed_output = reverse_augmentation(output, inverse_aug)
            augmented_outputs.append(reversed_output)

    # Convert predictions to NumPy array
    augmented_outputs = np.array(augmented_outputs)

    # Compute mean and variance across augmentations
    mean_output = np.mean(augmented_outputs, axis=0)

    softmax_output = torch.softmax(augmented_outputs, dim=0)
    variance_output = np.var(softmax_output, axis=0)

    return mean_output, variance_output


def tta_entropy(model_inferer, input_data, device, n_iterations=10):
    """
    Run test-time augmentation to estimate uncertainty (entropy-based).

    Args:
        model: Trained model with dropout layers.
        input_data: Input image tensor.
        model_inferer: Inference function (e.g., sliding window inference).
        n_iterations: Number of times to run inference with dropout active.

    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    augmentations = get_augmentations()  # Get list of augmentations with inverses
    augmented_outputs = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in tqdm(range(n_iterations), desc="Predicting with TTA.."):
            # Randomly pick an augmentation
            aug_entry = np.random.choice(augmentations)
            aug = aug_entry["aug"]
            inverse_aug = aug_entry["inverse"]

            # Apply augmentation
            augmented_input = apply_augmentation(input_data, aug).to(device)

            # Forward pass
            pred = model_inferer(augmented_input) #(240, 240, 155)
            output = torch.softmax(pred, dim=1).cpu().numpy()

            # Reverse augmentation
            reversed_output = reverse_augmentation(output, inverse_aug)
            augmented_outputs.append(reversed_output)

    # Convert predictions to NumPy array
    augmented_outputs = np.array(augmented_outputs)

    # Compute mean and variance across augmentations
    mean_output = np.mean(augmented_outputs, axis=0)
    mean_output = np.squeeze(mean_output, axis=0)
    mean_output = np.clip(mean_output, 1e-6, 1.0)

    # Compute entropy: -p * log(p)
    epsilon = 1e-6  # To avoid log(0)
    entropy_output = -np.sum(mean_output * np.log(mean_output + epsilon), axis=0)

    return mean_output, entropy_output


def visualize_augmentation(
    original_slice,
    augmented_slice,
    slice_index,
    save_path="./augmentation_visualizations",
):
    """
    Visualize the original and augmented slices side by side.
    """
    os.makedirs(save_path, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_slice, cmap="gray")
    axes[0].set_title(f"Original Slice {slice_index}")
    axes[0].axis("off")

    axes[1].imshow(augmented_slice, cmap="gray")
    axes[1].set_title(f"Augmented Slice {slice_index}")
    axes[1].axis("off")

    plt.savefig(os.path.join(save_path, f"augmentation_slice_{slice_index}.png"))
    plt.close(fig)
