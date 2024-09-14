import os
import shutil
import json
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subfiles, subdirs
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# Check and set nnUNet paths
if 'nnUNet_raw' not in os.environ or 'nnUNet_preprocessed' not in os.environ or 'nnUNet_results' not in os.environ:
    raise EnvironmentError("nnUNet environment variables are not set. Please set nnUNet_raw, nnUNet_preprocessed, and nnUNet_results.")

nnUNet_raw = os.environ['nnUNet_raw']
nnUNet_preprocessed = os.environ['nnUNet_preprocessed']
nnUNet_results = os.environ['nnUNet_results']

def copy_BraTS_segmentation_and_convert_labels_to_nnUNet(in_file: str, out_file: str) -> None:
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 3, 4]:
            raise RuntimeError('unexpected label')
    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

brats_data_dir = '/home/agata/Desktop/thesis_tumor_segmentation/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'

task_id = 137
task_name = "BraTS2024"
foldername = "Dataset%03.0d_%s" % (task_id, task_name)

# Setting up nnU-Net folders
out_base = os.path.join(nnUNet_raw, foldername)
imagestr = os.path.join(out_base, "imagesTr")
labelstr = os.path.join(out_base, "labelsTr")
maybe_mkdir_p(imagestr)
maybe_mkdir_p(labelstr)

case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

for c in case_ids:
    shutil.copy(os.path.join(brats_data_dir, c, c + "-t1n.nii.gz"), os.path.join(imagestr, c + '_0000.nii.gz'))
    shutil.copy(os.path.join(brats_data_dir, c, c + "-t1c.nii.gz"), os.path.join(imagestr, c + '_0001.nii.gz'))
    shutil.copy(os.path.join(brats_data_dir, c, c + "-t2f.nii.gz"), os.path.join(imagestr, c + '_0002.nii.gz'))
    shutil.copy(os.path.join(brats_data_dir, c, c + "-t2w.nii.gz"), os.path.join(imagestr, c + '_0003.nii.gz'))
    copy_BraTS_segmentation_and_convert_labels_to_nnUNet(
        os.path.join(brats_data_dir, c, c + "-seg.nii.gz"),
        os.path.join(labelstr, c + '.nii.gz')
    )

generate_dataset_json(out_base,
                      channel_names={0: 'T1', 1: 'T1Gd', 2: 'T2', 3: 'Flair'},
                      labels={
                          'background': 0,
                          'whole tumor': (1, 2, 3),
                          'tumor core': (2, 3),
                          'enhancing tumor': (3, )
                      },
                      num_training_cases=len(case_ids),
                      file_ending='.nii.gz',
                      regions_class_order=(1, 2, 3),
                      license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                      reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                      dataset_release='1.0')

print(f"dataset.json created in {out_base}")
