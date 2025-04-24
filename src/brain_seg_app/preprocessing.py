import os, subprocess
import SimpleITK as sitk

def skull_strip_with_hd_bet(input_path, output_path=None):
    """
    Run HD-BET skull stripping on a single NIfTI file.
    """
    if output_path is None:
        base = os.path.basename(input_path).replace(".nii", "").replace(".gz", "")
        # Ensure the output filename ends with .nii.gz
        output_path = os.path.join(os.getcwd(), f"{base}_skullstripped.nii.gz")
    
    try:
        subprocess.run([
            "hd-bet",
            "-i", input_path,
            "-o", output_path,
            "-device", "cuda"  # Change to "cpu" if needed
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[HD-BET ERROR] {e}")
        return input_path  # fallback: return original file path

    # Return the skull-stripped image file (the mask is saved separately)
    return output_path


def realign_images_to_reference(nifti_paths, output_dir, progress_callback=None):
    """
    Skull-strips and realigns all images in 'nifti_paths' to the first one (as reference).
    The preprocessed images are saved into the specified output_dir.
    Optionally updates a Streamlit progress bar.
    """
    if not nifti_paths:
        return []

    stripped_paths = []
    total_steps = len(nifti_paths) * 2  # Skull stripping + registration
    current_step = 0

    for p in nifti_paths:
        # Create output path in output_dir
        base_name = os.path.basename(p).replace(".nii", "").replace(".gz", "")
        stripped_output = os.path.join(output_dir, f"{base_name}_skullstripped.nii.gz")
        stripped = skull_strip_with_hd_bet(p, output_path=stripped_output)
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
