import numpy as np
import torch
import os
import config.config as config
import nibabel as nib
import matplotlib.pyplot as plt


def save_checkpoint(model, epoch, filename="model.pt", best_acc=0, dir_add=None):
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}

    # Construct the full file path
    filename = os.path.join(config.output_dir, filename)

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


def convert_to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()  # Convert tensor to list
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
