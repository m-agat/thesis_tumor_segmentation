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
from utils.utils import save_checkpoint, EarlyStopping
from train_pipeline import trainer 
import dataset.dataloaders as dataloaders 

# MONAI Library Imports
from monai.inferers import sliding_window_inference

# PyTorch Utilities
from torch.utils.tensorboard import SummaryWriter

def cross_validate_trainer(
    model_class,  # Pass the model class so that a new instance is created for each fold
    optimizer_func,
    loss_func,
    acc_func,
    scheduler_func,
    num_folds,
    post_activation=None,
    post_pred=None,
):
    fold_results = []  # To store performance metrics for each fold

    for fold in range(num_folds):
        print(f"\nStarting Fold {fold + 1}/{num_folds}")

        train_loader, val_loader = dataloaders.get_loaders(
            batch_size=config.batch_size, 
            json_path=config.json_path, 
            basedir=config.root_dir, 
            fold=fold, 
            roi=config.roi
        )

        print(f"Data loaders for fold {fold} loaded.\n")

        # Create a new log directory for each fold
        fold_log_dir = f"./outputs/runs/fold_{fold + 1}"
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

    print("fold res: ", fold_results)
    print("metrics his", metrics_history)
    # Compute average metrics across folds
    avg_metrics = {
    key: np.nanmean([fold[key][-1] if isinstance(fold[key], (list, np.ndarray)) else fold[key] for fold in fold_results])
        for key in fold_results[0]
        if key not in {"loss_epochs", "trains_epoch", "val_acc_max"}
    }

    avg_metrics["avg_dice"] = np.nanmean([fold["dice"] for fold in fold_results])
    avg_metrics["avg_hd95"] = np.nanmean([fold["hd95"] for fold in fold_results])
    avg_metrics["avg_sensitivity"] = np.nanmean([fold["sensitivity"] for fold in fold_results])
    avg_metrics["avg_specificity"] = np.nanmean([fold["specificity"] for fold in fold_results])

    print("\nCross-Validation Results:")
    for key, value in avg_metrics.items():
        print(f"Avg {key.replace('_', ' ').title()}: {value:.4f}")

    return fold_results, avg_metrics
