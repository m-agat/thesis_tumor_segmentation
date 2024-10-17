# train_stage2.py

import torch
import time
import config
import models.models as models
from losses import get_loss_function, dice_acc, post_softmax, post_pred
from utils.utils import save_checkpoint, EarlyStopping
from train_helpers import train_epoch, val_epoch
from patch_extraction import extract_patches_from_wt  # Patch extraction function
import numpy as np
from functools import partial
from monai.inferers import sliding_window_inference

# Load the pre-trained model from Stage 1
model = models.swinunetr_model
model.load_state_dict(torch.load("best_stage1_model.pth"))  # Load the best model from Stage 1

# Load the WT predictions from Stage 1
stage1_wt_predictions = torch.load("stage1_wt_predictions.pth")

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.config.max_epochs)

# Inference function
model_inferer = partial(
    sliding_window_inference,
    roi_size=config.global_roi,
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

# Main trainer function for Stage 2 (fine-tuning)
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
    post_softmax,
    post_pred,
    wt_predictions,  # WT predictions from Stage 1
    patch_size=(64, 64, 64),
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []

    # Fine-tuning: Use WT mask to extract smaller patches for substructure segmentation
    train_patches = extract_patches_from_wt(train_loader, wt_predictions, patch_size=patch_size)

    for epoch in range(start_epoch, config.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        # Train on extracted patches (ET, TC, WT)
        train_loss = train_epoch(
            model,
            train_patches,  # Use the extracted patches
            optimizer,
            epoch=epoch,
            loss_func=loss_func,  # Multi-class loss for ET, TC, WT
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

            # Validation on full image
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_softmax=post_softmax,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)

            print(f"Validation Epoch {epoch}: Dice_TC: {dice_tc}, Dice_WT: {dice_wt}, Dice_ET: {dice_et}")

            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)

            # Early stopping
            early_stopper(val_avg_acc, model, epoch, best_acc=val_acc_max)

            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}. Best accuracy: {val_acc_max}")
                break

            if val_avg_acc > val_acc_max:
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, best_acc=val_acc_max, filename=models.get_model_name(models.models_dict, model))

            scheduler.step()

    print("Training Stage 2 Finished! Best Accuracy: ", val_acc_max)
    return val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch

# Start training for Stage 2
start_epoch = 0

val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch = trainer(
    model=model,
    train_loader=config.train_loader,
    val_loader=config.val_loader,
    optimizer=optimizer,
    loss_func=get_loss_function(binary=False),  # Multi-class loss for ET, TC, WT
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_softmax=post_softmax,
    post_pred=post_pred,
    wt_predictions=stage1_wt_predictions,  # Use WT predictions from Stage 1
)
