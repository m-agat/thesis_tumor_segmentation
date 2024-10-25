import torch
import numpy as np

def enable_dropout(model):
    """
    Function to enable the dropout layers during inference.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def test_time_dropout(model, input_data, n_iterations=20):
    """
    Run test-time dropout to estimate uncertainty.
    
    Args:
        model: Trained model with dropout layers.
        input_data: Input image tensor (e.g., MRI scan).
        n_iterations: Number of times to run inference with dropout active.
        
    Returns:
        mean_output: Averaged prediction over all iterations.
        variance_output: Variance of predictions across iterations.
    """
    model.eval()  # Switch to evaluation mode
    enable_dropout(model)  # Enable dropout layers during inference
    
    # Collect predictions over multiple forward passes
    all_outputs = []
    with torch.no_grad():
        for _ in range(n_iterations):
            output = model(input_data)
            all_outputs.append(output.cpu().numpy())
    
    # Convert to NumPy array for easier manipulation
    all_outputs = np.array(all_outputs)
    
    # Compute mean and variance across iterations
    mean_output = np.mean(all_outputs, axis=0)
    variance_output = np.var(all_outputs, axis=0)
    
    return mean_output, variance_output
