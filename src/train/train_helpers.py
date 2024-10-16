import torch
import time
from utils.utils import AverageMeter
from monai.data import decollate_batch
import config.config as config

def train_epoch(model, loader, optimizer, epoch, loss_func):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        data, target = batch_data["image"].to(config.device), batch_data["label"].to(config.device)
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        run_loss.update(loss.item(), n=config.batch_size)
        print(
            f"Epoch {epoch}/{config.config.max_epochs} {idx}/{len(loader)}",
            f"loss: {run_loss.avg:.4f}",
            f"time {time.time() - start_time:.2f}s",
        )
        start_time = time.time()
    torch.cuda.empty_cache()
    return run_loss.avg

# Validation epoch function with probability maps
def val_epoch(model, loader, epoch, acc_func, model_inferer, post_activation, post_pred):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(config.device), batch_data["label"].to(config.device)
            logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            
            # Convert outputs to probability maps using softmax
            val_output_convert = [post_pred(post_activation(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            dice_tc, dice_wt, dice_et = run_acc.avg
            print(
                f"Val {epoch}/{config.max_epochs} {idx}/{len(loader)}",
                f", dice_tc: {dice_tc}, dice_wt: {dice_wt}, dice_et: {dice_et}",
                f", time {time.time() - start_time:.2f}s",
            )
            start_time = time.time()

    return run_acc.avg


def get_wt_predictions(model, data_loader, model_inferer):
    model.eval()  # Set the model to evaluation mode
    wt_predictions = []  # To store whole tumor predictions for all images

    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:  # Iterate over the batches from the data loader
            # Assuming the batch contains a dictionary with 'image' as the input
            inputs = batch['image'].to(model.device)  # Move inputs to the model's device (e.g., GPU)

            # Use the model_inferer for inference
            outputs = model_inferer(model, inputs)  # Apply the model inferer, could be sliding window or other

            # Extract the Whole Tumor (WT) class from the outputs
            # Assuming the output format has logits or probabilities, and WT corresponds to class 1
            wt_output = torch.argmax(outputs, dim=1)  # Get the class with the highest score

            # Post-processing if needed (for example, mapping to original size)
            # Append predictions to list
            wt_predictions.append(wt_output.cpu().numpy())  # Move to CPU and convert to numpy if needed

    return wt_predictions