import os
import torch
import time
import random
import numpy as np
import logging
import gc
from torch.backends import cudnn
from torch.utils.data import DataLoader
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Activations, AsDiscrete
from monai.inferers import sliding_window_inference
from utils.helpers import get_data_loaders, save_checkpoint, load_checkpoint
from utils.criteria import CombinedLoss
from models.unet_3d import UNet3D
from models.vnet import get_vnet
from models.resunetplusplus import ResUNetPlusPlus
from models.attention_unet import get_attention_unet
from models.segresnet import get_segresnet
from functools import partial
import yaml
from torch.cuda.amp import autocast, GradScaler 
import argparse

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from YAML file
def load_config(config_path="./conf/config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Initialize random seed for reproducibility
def init_random(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

# Load model based on configuration
def load_model(cfg):
    architecture = cfg['model']['architecture']
    in_channels = cfg['model']['in_channels']
    out_channels = cfg['model']['out_channels']

    if architecture == "unet3d":
        model = UNet3D(in_channels=in_channels, num_classes=out_channels)
    elif architecture == "vnet":
        model = get_vnet(in_channels=in_channels, out_channels=out_channels)
    elif architecture == "resunet_pp":
        model = ResUNetPlusPlus(in_channels=in_channels, out_channels=out_channels)
    elif architecture == "attention_unet":
        model = get_attention_unet(in_channels=in_channels, out_channels=out_channels)
    elif architecture == "segres_net":
        model = get_segresnet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model

# Train one epoch with gradient accumulation and mixed precision
def train_epoch(model, train_loader, optimizer, loss_func, device, scaler, accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch_data in enumerate(train_loader):
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        with autocast():
            outputs = model(inputs)
            loss = loss_func(outputs, labels) / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Validate model performance
def validate(model, val_loader, acc_func, model_inferer, post_sigmoid, post_pred, device):
    model.eval()
    acc_func.reset()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            outputs = model_inferer(inputs)
            outputs = [post_pred(post_sigmoid(output)) for output in decollate_batch(outputs)]
            labels = [post_pred(label) for label in decollate_batch(labels)]
            acc_func(y_pred=outputs, y=labels)

    avg_acc = acc_func.aggregate().item()
    
    torch.cuda.empty_cache()

    return avg_acc

# Train and validate the model
def train_and_validate(cfg, checkpoint_path=None):
    # Initialize random seeds and configure device
    print("Initializing random seed and device...")
    init_random(seed=cfg['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(cfg['data']['train_dir'], cfg['data']['val_dir'],
                                                batch_size=cfg['training']['batch_size'],
                                                num_workers=cfg['training']['num_workers'])
    print(f"Data loaded. Training samples: {len(train_loader)}, Validation samples: {len(val_loader)}")

    # Load model
    print(f"Loading model: {cfg['model']['architecture']}...")
    model = load_model(cfg)
    model.to(device)
    print("Model loaded successfully.")

    # Define loss function and optimizer
    print("Initializing loss function and optimizer...")
    loss_func = CombinedLoss(classes=cfg['model']['out_channels'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    print("Loss function and optimizer initialized.")

    # Define post-processing transforms and metrics
    post_sigmoid = Activations(sigmoid=True).to(device)  # Ensuring it's on the same device
    post_pred = AsDiscrete(threshold=0.5).to(device)
    acc_func = DiceMetric(include_background=True, reduction="mean_batch").to(device)

    # Initialize scaler for mixed precision training
    scaler = GradScaler()

    # Model inference using sliding window
    model_inferer = partial(sliding_window_inference, roi_size=cfg['model']['roi'], sw_batch_size=1, predictor=model, overlap=0.25)

    # Load checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, Best validation Dice: {best_acc:.4f}")

    # Training loop
    print(f"Starting training for {cfg['training']['epochs']} epochs...")
    for epoch in range(start_epoch, cfg['training']['epochs']):
        print(f"\n--- Epoch {epoch + 1}/{cfg['training']['epochs']} ---")
        logger.info(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")

        # Train for one epoch
        print("Training model...")
        train_loss = train_epoch(model, train_loader, optimizer, loss_func, device, scaler)
        logger.info(f"Training Loss: {train_loss:.4f}")
        print(f"Training Loss for epoch {epoch + 1}: {train_loss:.4f}")

        # Validate the model
        print("Validating model...")
        val_acc = validate(model, val_loader, acc_func, model_inferer, post_sigmoid, post_pred, device)
        logger.info(f"Validation Dice: {val_acc:.4f}")
        print(f"Validation Dice for epoch {epoch + 1}: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            print(f"New best model found with validation Dice {val_acc:.4f}, saving model...")
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, os.path.join(cfg['training']['checkpoint_dir'], 'best_model.pth'))

        # Save checkpoint every epoch
        save_checkpoint(model, optimizer, epoch, os.path.join(cfg['training']['checkpoint_dir'], f'checkpoint_{epoch + 1}.pth'))
        print(f"Checkpoint saved for epoch {epoch + 1}.")

        # Free GPU memory after each epoch
        torch.cuda.empty_cache()  
        print("Cleared GPU memory after epoch.")

        # Reset the accuracy function
        acc_func.reset()
        print("Accuracy function reset.")

    logger.info(f"Training complete. Best validation Dice: {best_acc:.4f}")
    print(f"Training complete. Best validation Dice: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and validate a segmentation model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint file to resume training from.')
    args = parser.parse_args()

    # Load configuration file
    cfg = load_config(args.config)

    # Create directories
    os.makedirs(cfg['training']['checkpoint_dir'], exist_ok=True)

    # Train and validate the model
    train_and_validate(cfg, checkpoint_path=args.checkpoint)

if __name__ == '__main__':
    main()
