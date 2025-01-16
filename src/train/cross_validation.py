# Core Libraries
import os
import sys
import json
import numpy as np
import torch
from functools import partial

# Add custom project modules to the path
sys.path.append("../")

# Project-Specific Imports
import config.config as config
import models.models as models
from utils.utils import save_checkpoint, EarlyStopping, convert_to_serializable
from train_pipeline import trainer 

# MONAI Library Imports
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.losses import GeneralizedDiceFocalLoss
from monai.transforms import AsDiscrete, Activations

# PyTorch Utilities
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)

# Loss 
loss_func = GeneralizedDiceFocalLoss(
    include_background=False, # We focus on subregions, not background
    to_onehot_y=False, # One-hot encoded in the transformations
    sigmoid=False, # Use softmax for multi-class segmentation
    softmax=True, # Multi-class softmax output
    w_type="square"
)

# Dice score
dice_acc = DiceMetric(
    include_background=False, 
    reduction=MetricReduction.MEAN_BATCH, # Compute average Dice for each batch
    get_not_nans=True,
)

# Optimizer and scheduler
def optimizer_func(params):
    return torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-5)

def scheduler_func(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

# Post-processing transforms
post_activation = Activations(softmax=True) # Softmax for multi-class output
post_pred = AsDiscrete(argmax=True, to_onehot=4) # get the class with the highest prob for each channel

def cross_validate_trainer(
    model_class,  # Pass the model class so that a new instance is created for each fold
    fold_loaders,
    optimizer_func,
    loss_func,
    acc_func,
    scheduler_func,
    num_folds,
    post_activation=None,
    post_pred=None,
):
    fold_results = []  # To store performance metrics for each fold

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\nStarting Fold {fold + 1}/{num_folds}")

        # Create a new log directory for each fold
        fold_log_dir = f"./outputs/runs/experiment_1/fold_{fold + 1}"
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_log_dir)  # Separate writer for each fold
        
        # Create a new model for each fold
        model = model_class().to(config.device)
        optimizer = optimizer_func(model.parameters())  
        scheduler = scheduler_func(optimizer) 

        model_inferer = partial(
            sliding_window_inference,
            roi_size=config.roi,
            sw_batch_size=config.sw_batch_size,
            predictor=model,  
            overlap=config.infer_overlap,
        )

        early_stopper = EarlyStopping(
            patience=5,
            delta=0.001,
            verbose=True,
            save_checkpoint_fn=save_checkpoint,
            filename=f"best_model_fold_{fold + 1}.pt",
        )

        # Run the trainer for this fold
        val_acc_max, metrics_history = trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_func=loss_func,
            acc_func=acc_func,
            scheduler=scheduler,
            model_inferer=model_inferer,
            start_epoch=0,
            post_activation=post_activation,
            post_pred=post_pred,
            early_stopper=early_stopper,
            fold=fold,
            writer=writer,
        )

        print(f"Finished Fold {fold + 1}, Best Dice Avg: {val_acc_max:.4f}")
        
        # Save fold results
        metrics_history["val_acc_max"] = val_acc_max
        fold_results.append(metrics_history)

        # Close the writer for this fold
        writer.close()

        torch.cuda.empty_cache()

    # Compute average metrics across folds
    avg_metrics = {
        key: np.nanmean([fold[key][-1] if isinstance(fold[key], (list, np.ndarray)) else fold[key] for fold in fold_results])
        for key in fold_results[0]
        if key not in {"loss_epochs", "trains_epoch", "val_acc_max"}
    }

    print("\nCross-Validation Results:")
    for key, value in avg_metrics.items():
        print(f"Avg {key.replace('_', ' ').title()}: {value:.4f}")

    return fold_results

# Perform Cross-Validation
fold_results = cross_validate_trainer(
    model_class=lambda: models.models_dict[f"{config.model_name}_model.pt"],
    fold_loaders=config.subset_fold_loaders, 
    optimizer_func=optimizer_func,
    loss_func=loss_func,
    acc_func=dice_acc,
    scheduler_func=scheduler_func,
    num_folds=config.num_folds,
    post_activation=post_activation,
    post_pred=post_pred
)

serializable_fold_results = convert_to_serializable(fold_results)
cv_results_path = os.path.join(config.output_dir, f"{config.model_name}_cv_results.json")
with open(cv_results_path, "w") as f:
    json.dump(serializable_fold_results, f, indent=4)

print(f"Cross-validation results saved to {cv_results_path}")

# tensorboard access: tensorboard --logdir=./outputs/runs/experiment_1/fold_1
