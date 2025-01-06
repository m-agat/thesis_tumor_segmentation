import sys
sys.path.append("../")
import os
import torch
import time
import config.config as config
import models.models as models
from utils.utils import save_checkpoint, EarlyStopping
from train_helpers import train_epoch, val_epoch
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.losses import GeneralizedDiceFocalLoss
from monai.transforms import AsDiscrete, Activations
from train import trainer

# Initialize the model
model = models.vnet_model
filename = models.get_model_name(models.models_dict, model)

print("Training, ", filename)

# Loss and accuracy
loss_func = GeneralizedDiceFocalLoss(
    include_background=False, # We focus on subregions, not background
    to_onehot_y=True, # Convert ground truth labels to one-hot encoding
    sigmoid=False, # Use softmax for multi-class segmentation
    softmax=True, # Multi-class softmax output
    w_type="square"
)

dice_acc = DiceMetric(
    include_background=False, 
    reduction=MetricReduction.MEAN_BATCH, # Compute average Dice for each batch
    get_not_nans=True
)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.max_epochs
)

# Post-processing transforms
post_activation = Activations(softmax=True) # Softmax for multi-class output
post_pred = AsDiscrete(argmax=True) # get the class with the highest prob

# Inference function
model_inferer = partial(
    sliding_window_inference,
    roi_size=config.roi,
    sw_batch_size=config.sw_batch_size,
    predictor=model,
    overlap=config.infer_overlap,
)

# Early stopping mechanism
early_stopper = EarlyStopping(
    patience=20,
    delta=0.001,
    verbose=True,
    save_checkpoint_fn=save_checkpoint,
    filename=models.get_model_name(models.models_dict, model),
)

# Start training
start_epoch = 0

def cross_validate_trainer(
    model_class,  # Pass the model class so that a new instance is created for each fold
    fold_loaders,
    optimizer_func,
    loss_func,
    acc_func,
    scheduler_func,
    model_inferer=None,
    num_folds=5,
    post_activation=None,
    post_pred=None,
    early_stopper=None,
):
    fold_results = []  # To store performance metrics for each fold

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\nStarting Fold {fold + 1}/{num_folds}")
        
        # Create a new model for each fold
        model = model_class().to(config.device)
        
        optimizer = optimizer_func(model.parameters())  
        scheduler = scheduler_func(optimizer) 

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
    model_class=model_class,
    fold_loaders=config.fold_loaders, 
    optimizer_func=optimizer_func,
    loss_func=loss_func,
    acc_func=dice_acc,
    scheduler_func=scheduler_func,
    model_inferer=model_inferer,
    num_folds=5,
    post_activation=post_activation,
    post_pred=post_pred,
)

with open("cv_results.json", "w") as f:
    json.dump(fold_results, f, indent=4)

print("Cross-validation results saved to 'cv_results.json'")