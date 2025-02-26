import sys 
import json 
import os 
import itertools
from functools import partial

sys.path.append("../")

# Custom
import config.config as config
import models.models as models
from train_pipeline import trainer
from utils.utils import save_checkpoint, EarlyStopping
from dataset import dataloaders

# MONAI
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.losses import GeneralizedDiceFocalLoss
from monai.transforms import AsDiscrete, Activations
from monai.inferers import sliding_window_inference

import torch
from torch.utils.tensorboard import SummaryWriter

# Enable cuDNN benchmark for optimized performance
torch.backends.cudnn.benchmark = True 

torch.manual_seed(42)     

# Loss 
loss_func = GeneralizedDiceFocalLoss(
    include_background=False, # We focus on subregions, not background
    to_onehot_y=False, # One-hot encoded in the transformations
    sigmoid=False, # Use softmax for multi-class segmentation
    softmax=True, # Multi-class softmax output
    w_type="square"
)

# Dice score
dice_acc = DiceMetric(
    include_background=False, 
    reduction=MetricReduction.MEAN_BATCH, # Compute average Dice for each batch
    get_not_nans=True,
)

# Post-processing transforms
post_activation = Activations(softmax=True) # Softmax for multi-class output
post_pred = AsDiscrete(argmax=True, to_onehot=4) # get the class with the highest prob for each channel

model_inferer = partial(
            sliding_window_inference,
            roi_size=config.roi,
            sw_batch_size=config.sw_batch_size,
            predictor=models.final_models_dict[f"{config.model_name}_model.pt"],  
            overlap=config.infer_overlap,
        )

early_stopper = EarlyStopping(
            patience=10,
            delta=0.001,
            verbose=True,
            save_checkpoint_fn=save_checkpoint,
            filename=f"best_{config.model_name}_model.pt",
        )

train_loader, val_loader = dataloaders.get_loaders(
            batch_size=config.batch_size, 
            json_path=config.json_path, 
            basedir=config.root_dir, 
            fold=None, 
            roi=config.roi,
            use_final_split=True
        )

lr = 0.0001
wd = 0.0001
optimizer = torch.optim.AdamW(models.final_models_dict[f"{config.model_name}_model.pt"].parameters(), lr=lr, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)

# Create a new log directory for each fold
fold_log_dir = os.path.join(config.output_dir, f"runs/FINAL_TRAINING")
os.makedirs(fold_log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=fold_log_dir)

# Start training
start_epoch = 0
print(f"Starting training {config.model_name} model")
val_acc_max, metrics_history = (trainer(
            model=models.final_models_dict[f"{config.model_name}_model.pt"],
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_func=loss_func,
            acc_func=dice_acc,
            scheduler=scheduler,
            model_inferer=model_inferer,
            start_epoch=start_epoch,
            post_activation=post_activation,
            post_pred=post_pred,
            early_stopper=early_stopper,
            use_folds=False,
            fold=None,
            writer=writer,
    )
)