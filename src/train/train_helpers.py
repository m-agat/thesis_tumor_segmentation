import torch
import time
import os
from utils.utils import AverageMeter
from monai.data import decollate_batch
import config.config as config
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_epoch(model, loader, optimizer, epoch, loss_func, writer=None, fold=0):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(config.device), batch_data["label"].to(
            config.device
        )

        # Zero the gradients before the forward pass
        optimizer.zero_grad()

        # Mixed precision: use autocast for forward pass and loss computation
        with autocast():
            logits = model(data)
            loss = loss_func(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss.update(loss.item(), n=config.batch_size)

        if writer:
            # Log batch-wise training loss
            writer.add_scalar(f"Fold_{fold + 1}/Batch_Loss/Train", loss.item(), epoch * len(loader) + idx)

        print(
            f"Epoch {epoch}/{config.max_epochs} {idx}/{len(loader)}",
            f"loss: {run_loss.avg:.4f}",
            f"time {time.time() - start_time:.2f}s",
        )

        start_time = time.time()

    # Log average training loss for the epoch
    if writer:
        writer.add_scalar(f"Fold_{fold + 1}/Loss/Train_Epoch", run_loss.avg, epoch)

    torch.cuda.empty_cache()

    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    loss_func,
    model_inferer=None,
    post_activation=None,
    post_pred=None,
    writer=None,
    fold=0,
):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    run_loss = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(config.device), batch_data[
                "label"
            ].to(config.device)
            logits = model_inferer(data)

            # Compute validation loss
            loss = loss_func(logits, target)
            run_loss.update(loss.item(), n=config.batch_size)

            # Compute Dice metrics
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [
                post_pred(post_activation(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            # Get Dice per subregion
            dice_ncr = run_acc.avg[0]
            dice_ed = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, config.max_epochs, idx, len(loader)),
                ", dice_ncr:",
                dice_ncr,
                ", dice_ed:",
                dice_ed,
                ", dice_et:",
                dice_et,
                ", loss: {:.4f}".format(run_loss.avg),
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    # Log average validation loss and Dice scores for the epoch
    if writer:
        writer.add_scalar(f"Fold_{fold + 1}/Loss/Validation_Epoch", run_loss.avg, epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_NCR", run_acc.avg[0], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_ED", run_acc.avg[1], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_ET", run_acc.avg[2], epoch)

    return run_acc.avg, run_loss.avg
