import os
import sys
import numpy as np
from skimage import filters
from skimage.filters import gabor
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.ndimage import center_of_mass
from scipy.stats import skew, kurtosis

# Feature Extraction Functions
def intensity_features(img_data, modality):
    """Extract intensity-based features: mean, standard deviation."""
    img_data_np = _to_numpy(img_data)
    return {
        f'{modality}_mean_intensity': np.mean(img_data_np),
        f'{modality}_stddev_intensity': np.std(img_data_np)
    }

def entropy_feature(img_data, modality):
    """Calculate entropy as a texture descriptor."""
    img_data_np = _to_numpy(img_data)
    return {f'{modality}_entropy': shannon_entropy(img_data_np)}

def edge_features(img_data, modality):
    """Calculate edge-based features using Sobel filter."""
    img_data_np = _to_numpy(img_data)
    sobel_edges = filters.sobel(img_data_np)
    return {
        f'{modality}_sobel_mean': np.mean(sobel_edges),
        f'{modality}_sobel_std': np.std(sobel_edges)
    }

def gabor_features(img_data, modality, frequencies=[0.1, 0.5, 0.8]):
    """Calculate texture features using Gabor filter for multiple frequencies."""
    img_data_np = _to_numpy(img_data)
    gabor_results = {}

    for freq in frequencies:
        gabor_magnitude_slices = []
        for i in range(img_data_np.shape[2]):  # Assuming axial slices along the last axis
            slice_2d = img_data_np[:, :, i]
            gabor_response, _ = gabor(slice_2d, frequency=freq)
            gabor_magnitude_slices.append(np.mean(gabor_response))
        gabor_results[f'{modality}_gabor_mean_freq_{freq}'] = np.mean(gabor_magnitude_slices)
        gabor_results[f'{modality}_gabor_std_freq_{freq}'] = np.std(gabor_magnitude_slices)

    return gabor_results

def lbp_features(img_data, modality, radius=1):
    """Extract Local Binary Pattern (LBP) features for each 2D slice in a 3D image."""
    img_data_np = _to_numpy(img_data).astype(np.uint8)
    lbp_n_points = 8 * radius
    lbp_slices_mean = []
    lbp_slices_std = []

    for i in range(img_data_np.shape[2]):
        slice_2d = img_data_np[:, :, i]
        lbp_image = local_binary_pattern(slice_2d, lbp_n_points, radius, method='uniform')
        lbp_slices_mean.append(np.mean(lbp_image))
        lbp_slices_std.append(np.std(lbp_image))

    return {
        f'{modality}_lbp_mean': np.mean(lbp_slices_mean),
        f'{modality}_lbp_std': np.std(lbp_slices_std)
    }

def center_of_mass_feature(img_data, modality):
    """Calculate center of mass of the tumor region."""
    img_data_np = _to_numpy(img_data)
    threshold = np.percentile(img_data_np, 95)
    tumor_region = img_data_np > threshold
    if np.any(tumor_region):
        com = center_of_mass(tumor_region)
        return {f'{modality}_center_of_mass': com}
    else:
        return {f'{modality}_center_of_mass': (np.nan, np.nan, np.nan)}

def statistical_moments(img_data, modality):
    """Calculate higher-order statistical moments (skewness and kurtosis)."""
    img_data_np = _to_numpy(img_data).flatten()
    return {
        f'{modality}_skewness': skew(img_data_np),
        f'{modality}_kurtosis': kurtosis(img_data_np)
    }

def glcm_features(img_data, modality):
    """Calculate Grey Level Co-occurrence Matrix (GLCM) features for each 2D slice."""
    img_data_np = _to_numpy(img_data).astype(np.uint8)
    glcm_slices_contrast = []

    for i in range(img_data_np.shape[2]):
        slice_2d = img_data_np[:, :, i]
        glcm = graycomatrix(slice_2d, [1], [0, np.pi / 2], levels=256, symmetric=True, normed=True)
        glcm_slices_contrast.append(graycoprops(glcm, 'contrast').mean())

    return {
        f'{modality}_glcm_contrast_mean': np.mean(glcm_slices_contrast),
        f'{modality}_glcm_contrast_std': np.std(glcm_slices_contrast)
    }

def gradient_magnitude_feature(img_data, modality):
    """Calculate gradient magnitude feature."""
    img_data_np = _to_numpy(img_data)
    gradient_magnitude = np.gradient(img_data_np)
    return {f'{modality}_gradient_magnitude_mean': np.mean(gradient_magnitude)}

def cross_modality_features(modality_images):
    """Compute features across all modalities for global insights."""
    stacked_image = np.mean(np.stack([_to_numpy(img) for img in modality_images.values()], axis=-1), axis=-1)

    threshold = np.percentile(stacked_image, 95)
    combined_tumor_region = stacked_image > threshold
    if np.any(combined_tumor_region):
        com = center_of_mass(combined_tumor_region)
    else:
        com = (np.nan, np.nan, np.nan)

    combined_entropy = shannon_entropy(stacked_image)

    return {
        'combined_center_of_mass': com,
        'combined_entropy': combined_entropy
    }

# Main Feature Extraction Function
def extract_features_from_tensor(image_tensor, modality_labels=['flair', 't1', 't1ce', 't2']):
    """Extract features from MRI modalities given as tensors or NumPy arrays."""
    features = {}
    modality_images = {modality: image_tensor[idx] for idx, modality in enumerate(modality_labels)}

    for modality, img_data in modality_images.items():
        features.update(intensity_features(img_data, modality))
        features.update(entropy_feature(img_data, modality))
        features.update(edge_features(img_data, modality))
        features.update(gabor_features(img_data, modality))
        features.update(lbp_features(img_data, modality))
        features.update(statistical_moments(img_data, modality))
        features.update(glcm_features(img_data, modality))
        features.update(gradient_magnitude_feature(img_data, modality))

    features.update(cross_modality_features(modality_images))
    return features

def extract_regional_features(image_tensor, masks, modality_labels=['flair', 't1', 't1ce', 't2']):
    """
    Extract features for specific regions (NCR, ED, ET) using pre-masked images.
    Args:
        image_tensor: The tensor containing the MRI image modalities.
        masks: A dictionary with region masks {'NCR': ncr_mask, 'ED': ed_mask, 'ET': et_mask}.
        modality_labels: List of modality names corresponding to tensor channels.

    Returns:
        A dictionary of features for each region and modality.
    """
    regional_features = {}

    for region_name, mask in masks.items():
        print(f"    Processing {region_name} region...")
        for mod_idx, modality in enumerate(modality_labels):
            print(f"        Processing {modality} image...")
            img_data = _to_numpy(image_tensor[mod_idx])  # Convert to NumPy if needed
            masked_img = img_data * mask  # Apply the mask while retaining the shape

            # Skip empty regions
            if not np.any(mask):
                print(f"Skipping empty region '{region_name}' for modality '{modality}'")
                for feature in [
                    'mean_intensity', 'stddev_intensity', 'entropy', 'sobel_mean', 'sobel_std',
                    'gabor_mean_freq_0.1', 'gabor_std_freq_0.1', 'gabor_mean_freq_0.5',
                    'gabor_std_freq_0.5', 'gabor_mean_freq_0.8', 'gabor_std_freq_0.8',
                    'lbp_mean', 'lbp_std', 'skewness', 'kurtosis', 'glcm_contrast_mean',
                    'glcm_contrast_std', 'gradient_magnitude_mean'
                ]:
                    regional_features[f'{modality}_{region_name}_{feature}'] = np.nan
                continue

            # Compute features for the masked region
            masked_img_with_channel = np.expand_dims(masked_img, axis=0)
            region_features = extract_features_from_tensor(masked_img_with_channel, [modality])

            # Prefix features with region and modality names
            for key, value in region_features.items():
                regional_features[f'{modality}_{region_name}_{key}'] = value

    return regional_features

# Utility to handle NumPy/PyTorch conversion
def _to_numpy(data):
    """Convert PyTorch tensor to NumPy array if needed."""
    if hasattr(data, 'cpu'):  # If PyTorch tensor
        return data.cpu().numpy()
    return data  # Already NumPy
