import torch
import time
import os
from utils.utils import AverageMeter
from monai.data import decollate_batch
import config.config as config
from torch.cuda.amp import autocast, GradScaler
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric

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
    run_hd95_meter = AverageMeter()
    run_sensitivity = AverageMeter() 
    run_specificity = AverageMeter()  

    # ConfusionMatrixMetric instance
    confusion_metric = ConfusionMatrixMetric(
        include_background=False,  
        metric_name=["sensitivity", "specificity"], 
        reduction="none",  
        compute_sample=False  
    )

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

            # Compute Hausdorff 95 Distance 
            hd95 = compute_hausdorff_distance(
                y_pred=torch.stack(val_output_convert),
                y=torch.stack(val_labels_list),
                include_background=False,
                distance_metric="euclidean",
                percentile=95  # For HD95
            )

            # Update HD95 meter for all subregions
            hd95 = hd95.squeeze(0)  
            run_hd95_meter.update(hd95.cpu().numpy(), n=config.batch_size)

            # Compute Sensitivity and Specificity
            confusion_metric.reset()
            confusion_metric(y_pred=val_output_convert, y=val_labels_list)
            sensitivity, specificity = confusion_metric.aggregate()

            # Update sensitivity and specificity meters
            sensitivity = sensitivity.squeeze(0)  # Shape becomes [3]
            specificity = specificity.squeeze(0)
            run_sensitivity.update(sensitivity.cpu().numpy(), n=config.batch_size)
            run_specificity.update(specificity.cpu().numpy(), n=config.batch_size)


            # Get Dice per subregion
            dice_ncr, dice_ed, dice_et = run_acc.avg
            hd95_ncr, hd95_ed, hd95_et = run_hd95_meter.avg
            sensitivity_ncr, sensitivity_ed, sensitivity_et = run_sensitivity.avg
            specificity_ncr, specificity_ed, specificity_et = run_specificity.avg
            print(
                f"Val {epoch}/{config.max_epochs} {idx}/{len(loader)}\n",
                f"Dice NCR: {dice_ncr:.4f}, Dice ED: {dice_ed:.4f}, Dice ET: {dice_et:.4f}\n",
                f"HD95 NCR: {hd95_ncr:.2f}, HD95 ED: {hd95_ed:.2f}, HD95 ET: {hd95_et:.2f}\n",
                f"Sensitivity NCR: {sensitivity_ncr:.4f}, ED: {sensitivity_ed:.4f}, ET: {sensitivity_et:.4f}\n",
                f"Specificity NCR: {specificity_ncr:.4f}, ED: {specificity_ed:.4f}, ET: {specificity_et:.4f}\n",
                f"loss: {run_loss.avg:.4f}, time: {time.time() - start_time:.2f}s\n",
            )
            start_time = time.time()

    # Log average validation loss and Dice scores for the epoch
    if writer:
        writer.add_scalar(f"Fold_{fold + 1}/Loss/Validation_Epoch", run_loss.avg, epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_NCR", run_acc.avg[0], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_ED", run_acc.avg[1], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Dice/Validation_ET", run_acc.avg[2], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/HD95/Validation_NCR", run_hd95_meter.avg[0], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/HD95/Validation_ED", run_hd95_meter.avg[1], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/HD95/Validation_ET", run_hd95_meter.avg[2], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Sensitivity/Validation_NCR", run_sensitivity.avg[0], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Sensitivity/Validation_ED", run_sensitivity.avg[1], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Sensitivity/Validation_ET", run_sensitivity.avg[2], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Specificity/Validation_NCR", run_specificity.avg[0], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Specificity/Validation_ED", run_specificity.avg[1], epoch)
        writer.add_scalar(f"Fold_{fold + 1}/Specificity/Validation_ET", run_specificity.avg[2], epoch)

    return run_acc.avg, run_loss.avg
