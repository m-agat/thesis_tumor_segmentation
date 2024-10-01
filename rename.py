import nibabel as nib
import os
import shutil
import SimpleITK as sitk

def resample_image_to_reference(input_image_path, reference_image_path, output_image_path):
    """
    Resample an image to match the shape and spacing of a reference image.
    """
    # Read the input and reference images
    input_image = sitk.ReadImage(input_image_path)
    reference_image = sitk.ReadImage(reference_image_path)

    # Set up the resampler to use the reference image's spacing and dimensions
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)

    # Perform the resampling
    resampled_image = resampler.Execute(input_image)

    # Save the resampled image
    sitk.WriteImage(resampled_image, output_image_path)
    print(f"Resampled image saved to {output_image_path}")

# flair_path = '/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_raw/ARE_test/imagesARE/ARE_0000.nii.gz'
# t1_path = '/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_raw/ARE_test/imagesARE/ARE_0001.nii.gz'
# t1gd_path = '/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_raw/ARE_test/imagesARE/ARE_0002.nii.gz'
# t2_path = '/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_raw/ARE_test/imagesARE/ARE_0003.nii.gz'

# Resample FLAIR to match T1
# resample_image_to_reference(flair_path, t1_path, flair_path)

# Resample T1GD to match T1
# resample_image_to_reference(t1gd_path, t1_path, t1gd_path)

# Resample T2 to match T1
# resample_image_to_reference(t2_path, t1_path, t2_path)

def convert_nii_to_niigz(nii_path):
    """
    Convert a .nii file to .nii.gz format if necessary.
    """
    if nii_path.endswith(".nii") and not nii_path.endswith(".nii.gz"):
        img = nib.load(nii_path)
        nii_gz_path = nii_path + ".gz"
        nib.save(img, nii_gz_path)
        print(f"Converted {nii_path} to {nii_gz_path}")
        return nii_gz_path
    return nii_path 

def rename_and_move_files_for_nnunet(input_folder, case_id, output_folder):
    """
    Rename the files for nnUNet compatibility, convert .nii to .nii.gz if necessary, 
    and move them to the specified output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    # mapping = {
    #     "skullStripped_biasCorrected_ARE_FLAIR.nii": f"{case_id}_0000.nii.gz",
    #     "skullStripped_biasCorrected_ARE_T1.nii": f"{case_id}_0001.nii.gz",
    #     "skullStripped_biasCorrected_ARE_T13DGD.nii": f"{case_id}_0002.nii.gz",
    #     "skullStripped_biasCorrected_ARE_T2.nii": f"{case_id}_0003.nii.gz"
    # }

    mapping = {
        "skullStripped_biasCorrected_VIGO_03_FLAIR_coregistered.nii.gz": f"{case_id}_0000.nii.gz",
        "skullStripped_biasCorrected_VIGO_03_T1GAD_coregistered.nii.gz": f"{case_id}_0001.nii.gz",
        "skullStripped_biasCorrected_VIGO_03_T2_coregistered.nii.gz": f"{case_id}_0002.nii.gz",
        "skullStripped_biasCorrected_VIGO_03_VOIFLAIR_coregistered.nii.gz": f"{case_id}_VOIFLAIR_0000.nii.gz",
        "skullStripped_biasCorrected_VIGO_03_VOIT1GAD_coregistered.nii.gz": f"{case_id}_VOIT1GAD_0001.nii.gz",
    }

    for old_name, new_name in mapping.items():
        old_path = os.path.join(input_folder, old_name)
        old_path = convert_nii_to_niigz(old_path)

        new_path = os.path.join(output_folder, new_name)
        
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Moved and renamed {old_path} to {new_path}")

input_folder = "/home/agata/Desktop/thesis_tumor_segmentation/data/braintumor_data/VIGO_03/coregistered"
output_folder = "/home/agata/Desktop/thesis_tumor_segmentation/nnUNet/nnUNet_raw/VIGO_03_test/imagesTr"
case_id = "VIGO_03"
rename_and_move_files_for_nnunet(input_folder, case_id, output_folder)
