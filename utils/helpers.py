import torch 
from torch.utils.data import DataLoader
from data_preprocessing.dataset import BraTSDataset
import os

def get_data_loaders(train_dir, val_dir, batch_size, num_workers, transform=None, target_shape=(160, 192, 128)):
    """
    Create training and validation data loaders for BraTS 2024.
    """
    # Create lists of subject directories
    train_subjects = [os.path.join(train_dir, d) for d in os.listdir(train_dir)]
    val_subjects = [os.path.join(val_dir, d) for d in os.listdir(val_dir)]

    # Initialize datasets
    train_dataset = BraTSDataset(train_subjects, transform=transform, target_shape=target_shape, load_segmentation=True)
    val_dataset = BraTSDataset(val_subjects, transform=transform, target_shape=target_shape, load_segmentation=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch
