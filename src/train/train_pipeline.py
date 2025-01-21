import torch
import time
import os
import numpy as np 
import sys
from torch.cuda.amp import autocast, GradScaler

# Custom modules
sys.path.append("../")
from utils.utils import AverageMeter, save_checkpoint
import config.config as config

# MONAI
from monai.data import decollate_batch
from monai.metrics import compute_hausdorff_distance, ConfusionMatrixMetric


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, writer=None, fold=0):
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

        if writer and idx % 10 == 0:
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

    del data, target, logits
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
    optimizer_name=None,
    initial_lr=None,
    weight_decay_value=None
):
    fold_dir = os.path.join(
        config.output_dir,
        f"fold_{fold + 1}_results_{optimizer_name}_lr_{initial_lr}_wd_{weight_decay_value}"
    )
    os.makedirs(fold_dir, exist_ok=True)

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
    
    with torch.no_grad(), autocast():
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

            # Convert predictions to binary values
            val_output_convert = [
                post_pred(post_activation(val_pred_tensor))
                for val_pred_tensor in val_outputs_list
            ]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            dice_scores = acc.cpu().numpy()
            for i, dice_score in enumerate(dice_scores):
                if not_nans[i] == 0:  # Tissue is absent in ground truth
                    pred_empty = torch.sum(torch.stack(val_output_convert)[:, i]).item() == 0
                    dice_scores[i] = 1.0 if pred_empty else 0.0
            run_acc.update(dice_scores, n=not_nans.cpu().numpy())

            # Compute Hausdorff 95 Distance 
            hd95 = compute_hausdorff_distance(
                y_pred=torch.stack(val_output_convert),
                y=torch.stack(val_labels_list),
                include_background=False,
                distance_metric="euclidean",
                percentile=95  # For HD95
            )

            # Update HD95 meter for all subregions
            hd95 = hd95.squeeze(0).cpu().numpy()  
            run_hd95_meter.update(hd95, n=config.batch_size)

            # Compute Sensitivity and Specificity
            confusion_metric.reset()
            confusion_metric(y_pred=val_output_convert, y=val_labels_list)
            sensitivity, specificity = confusion_metric.aggregate()

            # Update sensitivity and specificity meters
            sensitivity = sensitivity.squeeze(0).cpu().numpy()  # Shape becomes [3]
            specificity = specificity.squeeze(0).cpu().numpy()
            for i in range(len(sensitivity)):
                if not_nans[i] == 0:  # Tissue is absent
                    pred_empty = torch.sum(torch.stack(val_output_convert)[:, i]).item() == 0
                    sensitivity[i] = 1.0 if pred_empty else 0.0
                    specificity[i] = 1.0
            run_sensitivity.update(sensitivity, n=config.batch_size)
            run_specificity.update(specificity, n=config.batch_size)

            if epoch == config.max_epochs - 1:
                # Get probabilities
                probs = np.stack(
                    [post_activation(val_pred_tensor).cpu().numpy() for val_pred_tensor in val_outputs_list]
                )

                # Get ground truth class labels
                y_true_class = torch.argmax(torch.stack(val_labels_list), dim=1).cpu().numpy().flatten()

                # Get predicted class labels
                y_pred_class = torch.argmax(torch.stack(val_output_convert), dim=1).cpu().numpy().flatten()

                # Save files
                np.savez_compressed(os.path.join(fold_dir, f"probs_batch_{idx}.npz"), probs)
                np.savez_compressed(os.path.join(fold_dir, f"true_labels_batch_{idx}.npz"), y_true_class)
                np.savez_compressed(os.path.join(fold_dir, f"pred_labels_batch_{idx}.npz"), y_pred_class)

                del probs, y_true_class, y_pred_class

            # Get metrics per subregion
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

            del data, target, logits
            torch.cuda.empty_cache()

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

    return run_acc.avg, run_loss.avg, run_hd95_meter.avg, \
            run_sensitivity.avg, run_specificity.avg 

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
    writer=None,
):
    val_acc_max = 0.0
    val_loss_min = float("inf")
    metrics_history = {
        "dice_ncr": [],
        "dice_ed": [],
        "dice_et": [],
        "hd95_ncr": [],
        "hd95_ed": [],
        "hd95_et": [],
        "sensitivity_ncr": [],
        "sensitivity_ed": [],
        "sensitivity_et": [],
        "specificity_ncr": [],
        "specificity_ed": [],
        "specificity_et": [],
        "loss_epochs": [],
        "trains_epoch":[]
    }

    scaler = GradScaler()

    # Hyperparameters
    initial_lr = optimizer.param_groups[0]['lr']
    weight_decay_value = optimizer.param_groups[0]['weight_decay']
    optimizer_name = optimizer.__class__.__name__

    for epoch in range(start_epoch, config.max_epochs):
        print(time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        # Training epoch
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler=scaler,
            epoch=epoch,
            loss_func=loss_func,
            writer=writer,
            fold=fold
        )

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if writer:
            writer.add_scalar(f"Fold_{fold + 1}/LearningRate", current_lr, epoch)

        print(
            "Final training  {}/{}".format(epoch, config.max_epochs - 1),
            "loss: {:.4f}".format(train_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if (epoch + 1) % config.val_every == 0 or epoch == 0:
            metrics_history["loss_epochs"].append(train_loss)
            metrics_history["trains_epoch"].append(epoch)
            epoch_time = time.time()

            # Validation step
            val_acc, val_loss, hd95_scores, sensitivity_scores, specificity_scores = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                loss_func=loss_func,
                model_inferer=model_inferer,
                post_activation=post_activation,
                post_pred=post_pred,
                writer=writer,
                fold=fold,
                optimizer_name=optimizer_name,
                initial_lr=initial_lr,
                weight_decay_value=weight_decay_value
            )

            dice_ncr, dice_ed, dice_et = val_acc
            val_avg_acc = np.nanmean(val_acc)

            print(
                "Final validation stats {}/{}".format(epoch, config.max_epochs - 1),
                f", dice_ncr: {dice_ncr:.4f}, dice_ed: {dice_ed:.4f}, dice_et: {dice_et:.4f}",
                f", Dice_Avg: {val_avg_acc:.4f}",
                f", time: {time.time() - epoch_time:.2f}s",
            )

            # Store metrics for each subregion
            for key, values in zip(
                ["dice_ncr", "dice_ed", "dice_et"], val_acc
            ):
                metrics_history[key].append(values)

            for key, values in zip(
                ["hd95_ncr", "hd95_ed", "hd95_et"], hd95_scores
            ):
                metrics_history[key].append(values)

            for key, values in zip(
                ["sensitivity_ncr", "sensitivity_ed", "sensitivity_et"], sensitivity_scores
            ):
                metrics_history[key].append(values)

            for key, values in zip(
                ["specificity_ncr", "specificity_ed", "specificity_et"], specificity_scores
            ):
                metrics_history[key].append(values)

            if val_avg_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                save_checkpoint(model, epoch, best_acc=val_acc_max, filename=f"{optimizer_name}_lr_{initial_lr}_wd_{weight_decay_value}_best_acc_fold_{fold + 1}.pt")

            if val_loss < val_loss_min:
                print(f"new best loss ({val_loss_min:.6f} --> {val_loss:.6f}).")
                val_loss_min = val_loss
                save_checkpoint(model, epoch, best_acc=val_loss_min, filename=f"{optimizer_name}_lr_{initial_lr}_wd_{weight_decay_value}_bestloss_fold_{fold + 1}.pt")

            scheduler.step()

            # Check for early stopping
            if early_stopper is not None and epoch > 10: # trigger only after at least 10 epochs of training
                early_stopper(val_avg_acc, model, epoch, val_acc_max)
                if early_stopper.early_stop:
                    print("Early stopping triggered. Stopping training.")
                    break
    
    metrics_history["dice"] = np.nanmean([
        np.nanmean(metrics_history["dice_ncr"]),
        np.nanmean(metrics_history["dice_ed"]),
        np.nanmean(metrics_history["dice_et"]),
    ])
    metrics_history["hd95"] = np.nanmean([
        np.nanmean(metrics_history["hd95_ncr"]),
        np.nanmean(metrics_history["hd95_ed"]),
        np.nanmean(metrics_history["hd95_et"]),
    ])
    metrics_history["sensitivity"] = np.nanmean([
        np.nanmean(metrics_history["sensitivity_ncr"]),
        np.nanmean(metrics_history["sensitivity_ed"]),
        np.nanmean(metrics_history["sensitivity_et"]),
    ])
    metrics_history["specificity"] = np.nanmean([
        np.nanmean(metrics_history["specificity_ncr"]),
        np.nanmean(metrics_history["specificity_ed"]),
        np.nanmean(metrics_history["specificity_et"]),
    ])
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return val_acc_max, metrics_history
