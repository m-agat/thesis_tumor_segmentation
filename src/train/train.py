import sys
sys.path.append('../')
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
from monai.losses import GeneralizedDiceLoss
from monai.transforms import AsDiscrete, Activations

# Initialize the model
model = models.swinunetr_model
filename = models.get_model_name(models.models_dict, model)

# Loss and accuracy
loss_func = GeneralizedDiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=True,
            w_type="square"
        )
dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

post_activation = Activations(sigmoid=True)
post_pred = AsDiscrete(argmax=False, threshold=0.5)

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

# Main trainer function
def trainer(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    scheduler,
    model_inferer=None,
    start_epoch=0,
    post_sigmoid=None,
    post_pred=None,
    early_stopper=None,
):
    val_acc_max = 0.0
    dices_tc = []
    dices_wt = []
    dices_et = []
    dices_avg = []
    loss_epochs = []
    trains_epoch = []
    for epoch in range(start_epoch, config.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            epoch=epoch,
            loss_func=loss_func,
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
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, config.max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_wt.append(dice_wt)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)
            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(
                    model,
                    epoch,
                    best_acc=val_acc_max,
                    filename=filename
                )
            scheduler.step()

            # Check for early stopping
            if early_stopper is not None:
                early_stopper(val_avg_acc, model)
                if early_stopper.early_stop:
                    print("Early stopping triggered. Stopping training.")
                    break
                
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )

# Start training 
start_epoch = 0

val_acc_max, dices_tc, dices_wt, dices_et, dices_avg, loss_epochs, trains_epoch = trainer(
    model=model,
    train_loader=config.train_loader,
    val_loader=config.val_loader,
    optimizer=optimizer,
    loss_func=loss_func,  
    acc_func=dice_acc,
    scheduler=scheduler,
    model_inferer=model_inferer,
    start_epoch=start_epoch,
    post_sigmoid=post_activation,
    post_pred=post_pred,
)