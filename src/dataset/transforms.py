from monai import transforms
import torch

def prepare_wt_mask(label):
    """
    Convert a multi-class label (with 3 channels: NCR, ED, ET) into a 2-channel mask.
    Channel 0: Background (0 for tumor, 1 for background).
    Channel 1: Whole tumor (1 for any tumor region, 0 for background).
    """
    wt_mask = (label > 0).float().max(dim=0, keepdim=False)[0]
    background_mask = (label == 0).float().max(dim=0, keepdim=False)[0]
    
    # Combine into two-class mask
    two_class_mask = torch.stack([background_mask, wt_mask], dim=0)

    # Remove metadata if present
    if hasattr(label, 'meta'):
        two_class_mask.meta = label.meta
        # Adjust metadata to reflect binary class structure
        two_class_mask.meta['dim'] = [2, *two_class_mask.shape[1:]]
        two_class_mask.meta['original_channel_dim'] = 0  # Or remove this key if no longer relevant

    return two_class_mask


def train_transform_wt_binary(global_roi):
    global_transform_wt_binary = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                k_divisible=[global_roi[0], global_roi[1], global_roi[2]],
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=[global_roi[0], global_roi[1], global_roi[2]],
                random_size=False,
            ),
            transforms.Lambdad(keys="label", func=prepare_wt_mask),  # Prepare binary WT label
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=global_roi, random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    return global_transform_wt_binary


def val_transform_wt_binary():
    val_transform_wt_binary = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Lambdad(keys="label", func=prepare_wt_mask),  # Prepare binary WT label
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    return val_transform_wt_binary


def test_transform_wt_binary():
    test_transform_wt_binary = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.Lambdad(keys="label", func=prepare_wt_mask),  # Prepare binary WT label
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    return test_transform_wt_binary


def get_mc_transforms(local_roi):
    """
    Returns the transformations for global patches (larger patches).
    """
    mc_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "wt_mask"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.AddChanneld(keys="wt_mask"),  # Add WT mask as an additional input channel
            transforms.RandSpatialCropd(
                keys=["image", "label", "wt_mask"],
                roi_size=[local_roi[0], local_roi[1], local_roi[2]],
                random_size=False,
            ),
            transforms.RandFlipd(keys=["image", "label", "wt_mask"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label", "wt_mask"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label", "wt_mask"], prob=0.5, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    return mc_transform


def get_mc_val_transforms():
    """
    Returns the transformations for validation data (no augmentation, just normalization).
    """
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label", "wt_mask"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.AddChanneld(keys="wt_mask"),  
        ]
    )
    return val_transform


def get_test_transforms():
    """
    Returns the transformations for test data.
    """
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return test_transform
