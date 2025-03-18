import torch
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../")
import config.config as config


def enable_dropout(model):
    """
    Function to enable the dropout layers during inference.
    """
    for module in model.modules():
        if isinstance(
            module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
        ):
            module.train()


def ttd_variance(model, model_inferer, input_data, device, n_iterations=20):
    """
    Run test-time dropout to estimate uncertainty.

    Args:
        model: Trained model with dropout layers.
        input_data: Input image tensor.
        model_inferer: Inference function (e.g., sliding window inference).
        n_iterations: Number of times to run inference with dropout active.

    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    model.eval()  # Ensure model is in evaluation mode
    enable_dropout(model)  # Enable dropout layers

    all_outputs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in tqdm(range(n_iterations), desc="Predicting with TTD.."):
            output = model_inferer(input_data).to(device)
            all_outputs.append(output.cpu().numpy())

    # Convert to NumPy array for easier manipulation
    all_outputs = np.array(all_outputs)

    # Compute mean and variance across iterations
    mean_output = np.mean(all_outputs, axis=0)  # Shape: (240, 240, 155)
    variance_output = np.var(all_outputs, axis=0)

    return mean_output, variance_output


def ttd_entropy(model, input_data, model_inferer, device, n_iterations=20):
    """
    Run test-time dropout to estimate uncertainty (entropy-based).

    Args:
        model: Trained model with dropout layers.
        input_data: Input image tensor.
        model_inferer: Inference function (e.g., sliding window inference).
        n_iterations: Number of times to run inference with dropout active.

    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    model.eval()  # Ensure model is in evaluation mode
    enable_dropout(model)  # Enable dropout layers

    all_outputs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in tqdm(range(n_iterations), desc="Predicting with TTD.."):
            output = torch.sigmoid(model_inferer(input_data)).to(device)
            all_outputs.append(output.cpu().numpy())

    # Convert to NumPy array for easier manipulation
    all_outputs = np.array(all_outputs)

    # Compute mean and variance across iterations
    mean_output = np.mean(all_outputs, axis=0)
    mean_output = np.squeeze(mean_output, axis=0)

    # Compute entropy: -p * log(p)
    epsilon = 1e-6  # To avoid log(0)
    entropy_output = -np.sum(mean_output * np.log(mean_output + epsilon), axis=0)

    return mean_output, entropy_output


def minmax_uncertainties(uncertainty_map):
    # scale: 0-1
    min_val = np.min(uncertainty_map)
    max_val = np.max(uncertainty_map)

    # Avoid division by zero if max_val and min_val are the same
    if max_val - min_val == 0:
        return np.zeros_like(uncertainty_map)

    scaled_uncertainty_map = (uncertainty_map - min_val) / (max_val - min_val)
    return scaled_uncertainty_map
