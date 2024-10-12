from monai import transforms
import torch

def prepare_wt_mask(label):
    """
    Prepare a binary WT mask where all non-background labels are grouped into a single WT class.
    """
    wt_mask = torch.zeros_like(label)
    wt_mask[label > 0] = 1  # WT includes NCR, ED, ET (any non-background label is WT)
    return wt_mask


def train_transform_wt_binary(global_roi):
    global_transform_wt_binary = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
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
            transforms.AddChanneld(keys="wt_mask"),  # Add WT mask as an additional input channel
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
