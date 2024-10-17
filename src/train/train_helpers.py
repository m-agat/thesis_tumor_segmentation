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
            f"Epoch {epoch}/{config.max_epochs} {idx}/{len(loader)}",
            f"loss: {run_loss.avg:.4f}",
            f"time {time.time() - start_time:.2f}s",
        )
        start_time = time.time()
    torch.cuda.empty_cache()
    return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, model_inferer, post_activation, post_pred):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"].to(config.device), batch_data["label"].to(config.device)
            logits = model_inferer(data)
            # val_labels_list = decollate_batch(target)
            # val_outputs_list = decollate_batch(logits)

            val_labels_list = list(target)
            val_outputs_list = list(logits)

            val_output_convert = [post_pred(post_activation(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
            _, dice_wt = run_acc.avg 
            print(
                f"Val {epoch}/{config.max_epochs} {idx}/{len(loader)}",
                f", dice_wt: {dice_wt}",
                f", time {time.time() - start_time:.2f}s",
            )
            start_time = time.time()

    return run_acc.avg


def get_wt_predictions(model, data_loader, model_inferer):
    model.eval()  
    wt_predictions = []  

    with torch.no_grad():  
        for batch in data_loader:  
            inputs = batch['image'].to(config.device)  

            outputs = model_inferer(inputs)  

            # Extract the Whole Tumor (WT) class from the outputs
            wt_output = torch.argmax(outputs, dim=1) 

            wt_predictions.append(wt_output.cpu().numpy())  

    return wt_predictions