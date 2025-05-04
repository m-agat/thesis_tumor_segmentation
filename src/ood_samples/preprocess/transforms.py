import os 
import SimpleITK as sitk

def skull_strip_with_hd_bet(input_path, output_dir):
    """
    Run HD-BET skull stripping on a single NIfTI file.
    - input_path : full path to input volume (e.g. flair.nii.gz)
    - output_dir : directory in which to write results
    Returns path to the skull‑stripped image (.nii.gz).
    """
    import subprocess, os

    # make sure the folder exists
    os.makedirs(output_dir, exist_ok=True)

    # extract bare basename (no .nii or .nii.gz)
    base = os.path.basename(input_path)
    if base.endswith('.nii.gz'):
        base = base[:-7]
    elif base.endswith('.nii'):
        base = base[:-4]

    # build the two output files
    skull_file = os.path.join(output_dir, f"{base}_bet.nii.gz")
    mask_file  = os.path.join(output_dir, f"{base}_mask.nii.gz")

    # call hd-bet, outputting a single file;
    # --save_mask will also emit the mask_file
    subprocess.run([
        "hd-bet",
        "-i", input_path,
        "-o", skull_file,
        "-device", "cuda",
        "--disable_tta",
        "--save_bet_mask"
    ], check=True)

    # sanity‑check
    if not os.path.exists(skull_file):
        raise FileNotFoundError(f"HD-BET failed to write stripped image at {skull_file}")
    # if you care about the mask, you can also test mask_file here

    return skull_file


def preprocess_images(nifti_paths, output_dir, progress_callback=None):
    """
    Skull-strips and realigns all images in 'nifti_paths' to the first one (as reference).
    The preprocessed images are saved into the specified output_dir.
    Optionally updates a Streamlit progress bar.
    """
    if not nifti_paths:
        return []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    stripped_paths = []
    total_steps = len(nifti_paths) * 2  # Skull stripping + registration
    current_step = 0

    for p in nifti_paths:
        base_name = os.path.basename(p).replace(".nii", "").replace(".gz", "")
        # choose a _directory_ to hold HD‑BET's outputs for this volume
        this_out_dir = os.path.join(output_dir, base_name + "_hd-bet")
        stripped = skull_strip_with_hd_bet(p, output_dir=this_out_dir)

        stripped_paths.append(stripped)
        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    reference_path = stripped_paths[0]
    ref_img_sitk = sitk.ReadImage(reference_path, sitk.sitkFloat32)

    output_paths = []
    ref_output_path = os.path.join(output_dir, f"preproc_{os.path.basename(reference_path)}")
    sitk.WriteImage(ref_img_sitk, ref_output_path)
    output_paths.append(ref_output_path)

    for path in stripped_paths[1:]:
        mov_img_sitk = sitk.ReadImage(path, sitk.sitkFloat32)

        initial_transform = sitk.CenteredTransformInitializer(
            ref_img_sitk,
            mov_img_sitk,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMeanSquares()
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=40)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        final_transform = registration_method.Execute(
            sitk.Cast(ref_img_sitk, sitk.sitkFloat32),
            sitk.Cast(mov_img_sitk, sitk.sitkFloat32),
        )

        aligned_img = sitk.Resample(
            mov_img_sitk,
            ref_img_sitk,
            final_transform,
            sitk.sitkLinear,
            0.0,
            mov_img_sitk.GetPixelID(),
        )

        out_path = os.path.join(output_dir, f"preproc_{os.path.basename(path)}")
        sitk.WriteImage(aligned_img, out_path)
        output_paths.append(out_path)

        current_step += 1
        if progress_callback:
            progress_callback(current_step / total_steps)

    return output_paths