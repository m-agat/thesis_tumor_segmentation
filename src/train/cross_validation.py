# Core Libraries
import os
import sys
import numpy as np
import torch
from functools import partial
import time 

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
import gc 

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
    fold_times = []

    for fold in range(num_folds):
        print(f"\nStarting Fold {fold + 1}/{num_folds}")
        fold_start_time = time.time()

        train_loader, val_loader = dataloaders.get_loaders(
            batch_size=config.batch_size, 
            json_path=config.json_path, 
            basedir=config.root_dir, 
            fold=fold, 
            roi=config.roi
        )

        print(f"Data loaders for fold {fold} loaded.\n")
        
        # Create a new model for each fold
        print(f"Reinitializing model for Fold {fold + 1}/{num_folds}")
        del model
        torch.cuda.empty_cache()
        gc.collect()

        model = model_class().to(config.device)
        optimizer = optimizer_func(model.parameters())  
        scheduler = scheduler_func(optimizer) 

        # Hyperparameters
        initial_lr = optimizer.param_groups[0]['lr']
        weight_decay_value = optimizer.param_groups[0]['weight_decay']
        optimizer_name = optimizer.__class__.__name__

        # Create a new log directory for each fold
        fold_log_dir = os.path.join(config.output_dir, f"runs/{optimizer_name}_lr_{initial_lr}_wd_{weight_decay_value}_fold_{fold + 1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=fold_log_dir)  # Separate writer for each fold

        model_inferer = partial(
            sliding_window_inference,
            roi_size=config.roi,
            sw_batch_size=config.sw_batch_size,
            predictor=model,  
            overlap=config.infer_overlap,
        )

        early_stopper = EarlyStopping(
            patience=10,
            delta=0.001,
            verbose=True,
            save_checkpoint_fn=save_checkpoint,
            filename=f"{optimizer_name}_lr_{initial_lr}_wd_{weight_decay_value}_best_model_fold_{fold + 1}.pt",
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

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        fold_times.append(fold_duration)

        print(f"Finished Fold {fold + 1}, Best Dice Avg: {val_acc_max:.4f}")
        print(f"Fold {fold + 1} took {fold_duration:.2f} seconds.")
        
        # Save fold results
        metrics_history["val_acc_max"] = val_acc_max
        fold_results.append(metrics_history)

        # Close the writer for this fold
        writer.flush()
        writer.close()

        torch.cuda.empty_cache()

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

    # Compute total time for all folds
    total_time = sum(fold_times)
    avg_time_per_fold = total_time / num_folds

    print("\nCross-Validation Results:")
    for key, value in avg_metrics.items():
        print(f"Avg {key.replace('_', ' ').title()}: {value:.4f}")

    print(f"\nTotal Time for Cross-Validation: {total_time:.2f} seconds")
    print(f"Average Time per Fold: {avg_time_per_fold:.2f} seconds")

    return fold_results, avg_metrics
