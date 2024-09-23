import os
from monai import transforms, data

def load_test_data(data_dir, case_num):
    """
    Load test files for a given case number and apply transformations.
    """
    test_files = [
        {
            "image": [
                os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_flair.nii.gz"),
                os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t1ce.nii.gz"),
                os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t1.nii.gz"),
                os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_t2.nii.gz"),
            ],
            "label": os.path.join(data_dir, f"BraTS2021_{case_num}", f"BraTS2021_{case_num}_seg.nii.gz"),
        }
    ]

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    test_ds = data.Dataset(data=test_files, transform=test_transform)

    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    return test_loader