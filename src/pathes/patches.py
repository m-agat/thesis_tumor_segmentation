import numpy as np

def get_bounding_box(mask):
    """
    Get the bounding box of the tumor in the mask.
    The mask is expected to be a binary 3D numpy array.
    Returns the min and max coordinates for the bounding box.
    """
    non_zero_indices = np.where(mask > 0)
    z_min, y_min, x_min = np.min(non_zero_indices, axis=1)
    z_max, y_max, x_max = np.max(non_zero_indices, axis=1)

    return (z_min, z_max), (y_min, y_max), (x_min, x_max)


def extract_patch(image, mask, patch_size=(64, 64, 64)):
    """
    Extracts a 3D patch from the image centered around the tumor (WT) region.
    Uses the bounding box of the WT mask to define the region of interest.
    
    image: 3D numpy array of the image (e.g., FLAIR, T1, T1CE, T2)
    mask: 3D numpy array of the WT mask
    patch_size: tuple specifying the patch size (depth, height, width)
    
    Returns the extracted patch and the corresponding mask patch.
    """
    (z_min, z_max), (y_min, y_max), (x_min, x_max) = get_bounding_box(mask)
    
    # Center coordinates for the bounding box
    z_center = (z_min + z_max) // 2
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2

    # Define the patch size
    patch_depth, patch_height, patch_width = patch_size

    # Compute the start and end indices for the patch, ensuring the patch stays within bounds
    z_start = max(z_center - patch_depth // 2, 0)
    z_end = min(z_start + patch_depth, image.shape[0])

    y_start = max(y_center - patch_height // 2, 0)
    y_end = min(y_start + patch_height, image.shape[1])

    x_start = max(x_center - patch_width // 2, 0)
    x_end = min(x_start + patch_width, image.shape[2])

    # Extract the patch from the image and mask
    image_patch = image[z_start:z_end, y_start:y_end, x_start:x_end]
    mask_patch = mask[z_start:z_end, y_start:y_end, x_start:x_end]

    return image_patch, mask_patch

def extract_patches_from_wt(train_loader, wt_predictions, patch_size=(64, 64, 64)):
    """
    Extract patches from the training set using the predicted WT mask from Stage 1.
    
    train_loader: the original dataloader
    wt_predictions: the WT mask predictions from Stage 1
    patch_size: tuple specifying the patch size
    
    Returns a new dataloader that provides smaller patches centered on the WT region.
    """
    patches = []
    for i, (inputs, _) in enumerate(train_loader):
        wt_mask = wt_predictions[i]  # Get the WT mask from the Stage 1 predictions
        image_patch, mask_patch = extract_patch(inputs, wt_mask, patch_size)
        patches.append((image_patch, mask_patch))
    
    return patches
