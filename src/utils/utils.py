import numpy as np
import torch
import os
import config.config as config
import nibabel as nib

def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    # Use the outputs directory for saving in Azure ML environment
    if dir_add is None:
        dir_add = "./outputs"  # Azure ML outputs directory

    # Create directory if it doesn't exist
    os.makedirs(dir_add, exist_ok=True)

    # Construct the full file path
    filename = os.path.join(dir_add, filename)

    # Save the checkpoint
    torch.save(save_dict, filename)
    print("Saving checkpoint:", filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


class EarlyStopping:
    """Early stops the training if validation loss/accuracy doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        delta=0.001,
        verbose=True,
        save_checkpoint_fn=save_checkpoint,
        filename="model.pt",
    ):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 10
            delta (float): Minimum change to qualify as an improvement.
                            Default: 0.001
            verbose (bool): If True, prints a message for each validation improvement.
                            Default: True
            save_checkpoint_fn (callable): Function to call for saving checkpoints.
                            Default: None
            filename (str): Name of the checkpoint file.
                            Default: "model.pt"
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_checkpoint_fn = save_checkpoint_fn  # Use your save function
        self.filename = filename

    def __call__(self, val_score, model, epoch, best_acc):
        score = (
            -val_score
        )  # Assume higher is better, change to loss if you are tracking loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint_fn(
                model, epoch, filename=self.filename, best_acc=best_acc
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint_fn(
                model, epoch, filename=self.filename, best_acc=best_acc
            )
            self.counter = 0


def calculate_sensitivity(pred, true):
    true_positives = (pred * true).sum(dim=(2, 3, 4))
    false_negatives = ((1 - pred) * true).sum(dim=(2, 3, 4))
    sensitivity = true_positives / (
        true_positives + false_negatives + 1e-7
    )  # Avoid division by zero
    return sensitivity.mean().item()


def calculate_specificity(pred, true):
    true_negatives = ((1 - pred) * (1 - true)).sum(dim=(2, 3, 4))
    false_positives = (pred * (1 - true)).sum(dim=(2, 3, 4))
    specificity = true_negatives / (true_negatives + false_positives + 1e-7)
    return specificity.mean().item()

def save_nifti(data, affine, filename):
    """
    Save the given data as a NIfTI file.
    
    Args:
        data (np.ndarray): 3D or 4D array to be saved as NIfTI.
        affine (np.ndarray): Affine matrix for spatial orientation.
        filename (str): Path where to save the NIfTI file.
    """
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, filename)
    print(f"Saved segmentation to {filename}")