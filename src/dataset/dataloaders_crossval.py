import os
from monai import data
from dataset.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)
from monai.apps import CrossValidation
from monai.data import DataLoader, Dataset
from abc import ABC, abstractmethod
from monai.data import CacheDataset
import sys

def read_data_from_folders(train_folder):
    """
    Read all data (train + val) from the training folder.
    """
    def load_cases_from_folder(folder):
        cases = [f for f in os.listdir(folder) if f.startswith("BraTS2021")]
        files = []
        for case in cases:
            files.append(
                {
                    "image": [
                        os.path.join(folder, case, f"{case}_flair.nii.gz"),
                        os.path.join(folder, case, f"{case}_t1ce.nii.gz"),
                        os.path.join(folder, case, f"{case}_t1.nii.gz"),
                        os.path.join(folder, case, f"{case}_t2.nii.gz"),
                    ],
                    "label": os.path.join(folder, case, f"{case}_seg.nii.gz"),
                    "path": case,
                }
            )
        return files

    # Load all files for cross-validation
    all_files = load_cases_from_folder(train_folder)
    return all_files

class CVDataset(ABC, Dataset):
    """
    Base class to generate cross validation datasets.

    """

    def __init__(
        self,
        data,
        transform,
    ) -> None:
        data = self._split_datalist(datalist=data)
        Dataset.__init__(
            self, data, transform
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")
    

def create_cross_validation_datasets(data_list, num_folds, train_transform, val_transform):
    """
    Create datasets for cross-validation using MONAI's CrossValidation class.
    """
    cvdataset = CrossValidation(
        dataset_cls=CVDataset,  
        data=data_list,
        nfolds=num_folds,
        seed=42,  
        transform=train_transform
    )

    # Create train and validation datasets for each fold
    folds = list(range(num_folds))
    train_datasets = [cvdataset.get_dataset(folds=folds[:i] + folds[i + 1:]) for i in folds]
    val_datasets = [cvdataset.get_dataset(folds=[i], transform=val_transform) for i in folds]

    return train_datasets, val_datasets

def get_data_loaders_for_folds(train_datasets, val_datasets, batch_size):
    fold_loaders = []
    for i in range(len(train_datasets)):
        train_loader = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_datasets[i], batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        fold_loaders.append((train_loader, val_loader))
    return fold_loaders