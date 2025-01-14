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

# Initialize the model
# model = models.vnet_model
# filename = models.get_model_name(models.models_dict, model)

# print("Training, ", filename)

# Loss and accuracy
# loss_func = GeneralizedDiceFocalLoss(
#     include_background=False, # We focus on subregions, not background
#     to_onehot_y=True, # Convert ground truth labels to one-hot encoding
#     sigmoid=False, # Use softmax for multi-class segmentation
#     softmax=True, # Multi-class softmax output
#     w_type="square"
# )

# dice_acc = DiceMetric(
#     include_background=False, 
#     reduction=MetricReduction.MEAN_BATCH, # Compute average Dice for each batch
#     get_not_nans=True
# )

# # Optimizer and scheduler
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=config.max_epochs
# )

# # Post-processing transforms
# post_activation = Activations(softmax=True) # Softmax for multi-class output
# post_pred = AsDiscrete(argmax=True) # get the class with the highest prob

# # Inference function
# model_inferer = partial(
#     sliding_window_inference,
#     roi_size=config.roi,
#     sw_batch_size=config.sw_batch_size,
#     predictor=model,
#     overlap=config.infer_overlap,
# )

# # Early stopping mechanism
# early_stopper = EarlyStopping(
#     patience=20,
#     delta=0.001,
#     verbose=True,
#     save_checkpoint_fn=save_checkpoint,
#     filename=models.get_model_name(models.models_dict, model),
# )

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
    post_activation=None,
    post_pred=None,
    early_stopper=None,
    fold=0,
):
    val_acc_max = 0.0
    val_loss_min = float("inf")
    dices_ncr = []
    dices_ed = []
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
            val_acc, val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                post_activation=post_activation,
                post_pred=post_pred,
            )
            dice_ncr = val_acc[0]
            dice_ed = val_acc[1]
            dice_et = val_acc[2]
            val_avg_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch, config.max_epochs - 1),
                ", dice_ncr:",
                dice_ncr,
                ", dice_ed:",
                dice_ed,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_avg_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_ncr.append(dice_ncr)
            dices_ed.append(dice_ed)
            dices_et.append(dice_et)
            dices_avg.append(val_avg_acc)

            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, best_acc=val_acc_max, filename=f"model_best_acc_fold_{fold + 1}.pt")

            if val_loss < val_loss_min:
                print(f"new best loss ({val_loss_min:.6f} --> {val_loss:.6f}).")
                val_loss_min = val_loss
                save_checkpoint(model, epoch, best_acc=val_loss_min, filename=f"model_bestloss_fold_{fold + 1}.pt")

            scheduler.step()

            # Check for early stopping
            if early_stopper is not None:
                early_stopper(val_avg_acc, model, epoch, val_acc_max)
                if early_stopper.early_stop:
                    print("Early stopping triggered. Stopping training.")
                    break

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return (
        val_acc_max,
        dices_ncr,
        dices_ed,
        dices_et,
        dices_avg,
        loss_epochs,
        trains_epoch,
    )


# Start training
# start_epoch = 0

# val_acc_max, dices_ncr, dices_ed, dices_et, dices_avg, loss_epochs, trains_epoch = (
#     trainer(
#         model=model,
#         train_loader=config.train_loader,
#         val_loader=config.val_loader,
#         optimizer=optimizer,
#         loss_func=loss_func,
#         acc_func=dice_acc,
#         scheduler=scheduler,
#         model_inferer=model_inferer,
#         start_epoch=start_epoch,
#         post_activation=post_activation,
#         post_pred=post_pred,
#     )
# )
