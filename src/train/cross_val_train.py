import sys
sys.path.append("../")
import os
import torch
import time
import config.config as config
import models.models as models
from utils.utils import save_checkpoint, EarlyStopping, convert_to_serializable
from train_helpers import train_epoch, val_epoch
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.losses import GeneralizedDiceFocalLoss
from monai.transforms import AsDiscrete, Activations
from train import trainer
import json 
from torch.utils.tensorboard import SummaryWriter

# Loss and accuracy
loss_func = GeneralizedDiceFocalLoss(
    include_background=False, # We focus on subregions, not background
    to_onehot_y=False, # One-hot encoded in the transformations
    sigmoid=False, # Use softmax for multi-class segmentation
    softmax=True, # Multi-class softmax output
    w_type="square"
)

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
        val_acc_max, dices_ncr, dices_ed, dices_et, dices_avg, loss_epochs, trains_epoch = trainer(
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
        fold_results.append({
            "fold": fold + 1,
            "val_acc_max": val_acc_max,
            "dices_ncr": dices_ncr,
            "dices_ed": dices_ed,
            "dices_et": dices_et,
            "dices_avg": dices_avg,
        })

        # Close the writer for this fold
        writer.close()

    # Calculate average performance across folds
    avg_dice_ncr = np.mean([r["dices_ncr"][-1] for r in fold_results])
    avg_dice_ed = np.mean([r["dices_ed"][-1] for r in fold_results])
    avg_dice_et = np.mean([r["dices_et"][-1] for r in fold_results])
    avg_dice_total = np.mean([r["val_acc_max"] for r in fold_results])

    print(f"\nCross-Validation Results:")
    print(f"Avg Dice NCR: {avg_dice_ncr:.4f}, ED: {avg_dice_ed:.4f}, ET: {avg_dice_et:.4f}, Total Avg Dice: {avg_dice_total:.4f}")

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

with open(os.path.join(config.output_dir, "cv_results.json"), "w") as f:
    json.dump(serializable_fold_results, f, indent=4)

print("Cross-validation results saved to 'cv_results.json'")