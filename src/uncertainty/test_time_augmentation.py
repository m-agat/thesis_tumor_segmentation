from monai import transforms
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from monai.data.meta_tensor import MetaTensor
from copy import deepcopy


class ReversibleScaleIntensityd:
    def __init__(self, keys, factor):
        self.keys = keys
        self.factor = factor

    def __call__(self, data):
        data["image"] = data["image"] * (1 + self.factor)
        return data


class InverseReversibleScaleIntensityd:
    def __init__(self, keys, factor):
        self.keys = keys
        self.factor = factor

    def __call__(self, data):
        data["image"] = data["image"] / (1 + self.factor)
        return data


def get_augmentations():
    """
    Define reversible augmentations for TTA.
    """
    augmentations = [
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=0),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=0),
        },
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=1),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=1),
        },
        {
            "aug": transforms.Flipd(keys=["image"], spatial_axis=2),
            "inverse": transforms.Flipd(keys=["image"], spatial_axis=2),
        },
        {
            "aug": ReversibleScaleIntensityd(keys=["image"], factor=0.1),
            "inverse": InverseReversibleScaleIntensityd(keys=["image"], factor=0.1),
        },
        {
            "aug": transforms.ShiftIntensityd(keys="image", offset=0.1),
            "inverse": transforms.ShiftIntensityd(keys="image", offset=-0.1),
        },
    ]
    return augmentations


def apply_augmentation(image, aug):
    # image is a tensor: (1, C, D, H, W)
    meta_image = MetaTensor(image[0].clone(), meta={"spatial_shape": image[0].shape[1:]})
    data = {"image": meta_image}
    augmented_data = deepcopy(aug)(data)  # clone the transform to avoid metadata overwrite
    augmented_image = augmented_data["image"].unsqueeze(0)  # Add batch dim back
    return augmented_image, meta_image.meta  # return metadata too


def reverse_augmentation(output, inverse_aug, original_meta):
    """
    Reverse the augmentation applied to the output.
    """
    output_tensor = MetaTensor(torch.tensor(output[0]), meta=original_meta)
    data = {"image": output_tensor}
    reversed_data = deepcopy(inverse_aug)(data)
    reversed_output = reversed_data["image"].unsqueeze(0).numpy()
    return reversed_output


def tta_variance(model_inferer, input_data, device, n_iterations=10, on_step=None):
    """
    Run test-time augmentation with reversible transformations for uncertainty estimation.
    """
    augmentations = get_augmentations()
    augmented_outputs = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in tqdm(range(n_iterations), desc="Predicting with TTA.."):
            if on_step:
                on_step(i + 1, n_iterations)

            aug_entry = np.random.choice(augmentations)
            aug = aug_entry["aug"]
            inverse_aug = aug_entry["inverse"]

            # Apply augmentation
            augmented_input, meta = apply_augmentation(input_data, aug)
            augmented_input = augmented_input.to(device)

            # Forward pass
            pred = model_inferer(augmented_input)
            output = pred.cpu().numpy()

            # Reverse augmentation
            reversed_output = reverse_augmentation(output, inverse_aug, meta)
            augmented_outputs.append(reversed_output)

    augmented_outputs = np.array(augmented_outputs)
    mean_output = np.mean(augmented_outputs, axis=0)

    all_outputs_tensor = torch.from_numpy(augmented_outputs).float()
    softmax_outputs = torch.nn.functional.softmax(all_outputs_tensor, dim=0)
    variance_output = np.var(softmax_outputs.cpu().numpy(), axis=0)

    return mean_output, variance_output
