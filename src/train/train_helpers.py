import torch
import time
import os
from utils.utils import AverageMeter
from monai.data import decollate_batch
import config.config as config
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

scaler = GradScaler()
log_dir = "./outputs/runs/experiment_1"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

def train_epoch(model, loader, optimizer, epoch, loss_func):
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

        print(
            f"Epoch {epoch}/{config.max_epochs} {idx}/{len(loader)}",
            f"loss: {run_loss.avg:.4f}",
            f"time {time.time() - start_time:.2f}s",
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", run_loss.avg, epoch * len(loader) + idx)

        start_time = time.time()

    torch.cuda.empty_cache()

    # Log final averages for the epoch
    writer.add_scalar("Loss_epoch/train", run_loss.avg, epoch)

    return run_loss.avg


def val_epoch(
    model,
    loader,
    epoch,
    acc_func,
    loss_func,
    model_inferer=None,
    post_sigmoid=None,
    post_pred=None,
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
                post_pred(post_sigmoid(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            # Get Dice per subregion
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, config.max_epochs, idx, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", loss: {:.4f}".format(run_loss.avg),
                ", time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    # Log final averaged metrics to TensorBoard
    writer.add_scalar("Loss_epoch/val", run_loss.avg, epoch)
    writer.add_scalar("Dice_tc/val", dice_tc, epoch)
    writer.add_scalar("Dice_wt/val", dice_wt, epoch)
    writer.add_scalar("Dice_et/val", dice_et, epoch)

    return run_acc.avg, run_loss.avg
