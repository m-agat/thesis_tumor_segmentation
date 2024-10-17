import sys
sys.path.append('../')
import os
import torch
import time
import config.config as config
import models.models as models
from losses import get_loss_function, get_activation_function 
from utils.utils import save_checkpoint, EarlyStopping
from train_helpers import train_epoch, val_epoch, get_wt_predictions
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

# Initialize the model
model = models.swinunetr_model_stage1
filename = models.get_model_name(models.models_stage1, model)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

post_activation, post_pred = get_activation_function(binary_segmentation=True)  # Sigmoid activation for binary segmentation

dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

# Inference function
model_inferer = partial(
    sliding_window_inference,
    roi_size= [config.global_roi[0], config.global_roi[1], config.global_roi[2]],
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
    filename=filename,
)

# Main trainer function for Stage 1
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer,
    start_epoch,
    post_activation,
    post_pred,
):
    val_acc_max = 0.0
    dices_wt = []
    loss_epochs = []
    trains_epoch = []

    for epoch in range(start_epoch, config.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        # Train for WT vs Background
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,  # Binary loss for WT vs Background
        )
        print(
            "Final training  {}/{}".format(epoch, config.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % config.val_every == 0 or epoch == 0:
            loss_epochs.append(train_loss)
            trains_epoch.append(int(epoch))
            epoch_time = time.time()

            # Validation for WT vs Background
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_activation=post_activation,
                post_pred=post_pred,
            )
            dice_wt = val_acc[1]  # Dice for WT

            print(f"Validation Epoch {epoch}: Dice WT: {dice_wt}")

            # Early stopping
            early_stopper(val_acc[1], model, epoch, best_acc=val_acc_max)

            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best WT Dice: {val_acc_max}")
                break

            if val_acc[1] > val_acc_max:
                val_acc_max = val_acc[1]
                save_checkpoint(model, epoch, best_acc=val_acc_max, filename=filename)

            scheduler.step()

    print("Training Stage 1 Finished! Best WT Dice: ", val_acc_max)
    return val_acc_max, dices_wt, loss_epochs, trains_epoch

# Start training for Stage 1
start_epoch = 0

val_acc_max, dices_wt, loss_epochs, trains_epoch = trainer(
    model=model,
    train_loader=config.train_loader,
    val_loader=config.val_loader,
    optimizer=optimizer,
    loss_func=get_loss_function(binary_segmentation=True),  # Binary loss for WT vs Background
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_activation=post_activation,
    post_pred=post_pred,
)

# Save WT predictions for fine-tuning in Stage 2
stage1_wt_predictions = get_wt_predictions(model, config.train_loader_subset, model_inferer) 

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)  
checkpoint_path = os.path.join(output_dir, "swinunetr_stage1.pt")

torch.save(stage1_wt_predictions, checkpoint_path)