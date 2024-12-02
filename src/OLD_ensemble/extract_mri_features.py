import os
import sys
import gc
import numpy as np
import pandas as pd
import psutil

# Local modules
sys.path.append("../")
import config.config as config
from mri_feature_extraction import extract_features_from_tensor

total_patients = len(config.val_loader)

# Prepare output path
results_save_path = os.path.join(config.output_dir, "mri_features.csv")

# Initialize the CSV with headers
headers = ['Patient']
for mod in ["flair", "t1", "t1ce", "t2"]:
    headers.extend([
        f'{mod}_mean_intensity', f'{mod}_stddev_intensity', f'{mod}_entropy',
        f'{mod}_sobel_mean', f'{mod}_sobel_std',
        f'{mod}_gabor_mean_freq_0.1', f'{mod}_gabor_std_freq_0.1',
        f'{mod}_gabor_mean_freq_0.5', f'{mod}_gabor_std_freq_0.5',
        f'{mod}_gabor_mean_freq_0.8', f'{mod}_gabor_std_freq_0.8',
        f'{mod}_lbp_mean', f'{mod}_lbp_std',
        f'{mod}_skewness', f'{mod}_kurtosis',
        f'{mod}_glcm_contrast_mean', f'{mod}_glcm_contrast_std',
        f'{mod}_gradient_magnitude_mean'
    ])
# Add cross-modality features
headers.extend([
    'combined_center_of_mass_x', 'combined_center_of_mass_y', 'combined_center_of_mass_z',
    'combined_entropy'
])

# Save headers to the CSV
df = pd.DataFrame(columns=headers)
df.to_csv(results_save_path, index=False)

# Process each patient and save features
total_patients = len(config.val_loader)

for idx, batch_data in enumerate(config.val_loader):
    image = batch_data["image"].to(config.device)
    patient_path = batch_data["path"]
    patient_id = os.path.basename(patient_path[0])  # Extract patient ID from path

    print(f"Processing patient {idx + 1}/{total_patients}: {patient_id}")

    # Extract features
    mri_features = extract_features_from_tensor(image[0])
    feature_vector = [patient_id]  # Start with the patient ID

    for mod in ["flair", "t1", "t1ce", "t2"]:
        feature_vector.extend([
            mri_features[f'{mod}_mean_intensity'],
            mri_features[f'{mod}_stddev_intensity'],
            mri_features[f'{mod}_entropy'],
            mri_features[f'{mod}_sobel_mean'],
            mri_features[f'{mod}_sobel_std'],
            mri_features[f'{mod}_gabor_mean_freq_0.1'],
            mri_features[f'{mod}_gabor_std_freq_0.1'],
            mri_features[f'{mod}_gabor_mean_freq_0.5'],
            mri_features[f'{mod}_gabor_std_freq_0.5'],
            mri_features[f'{mod}_gabor_mean_freq_0.8'],
            mri_features[f'{mod}_gabor_std_freq_0.8'],
            mri_features[f'{mod}_lbp_mean'],
            mri_features[f'{mod}_lbp_std'],
            mri_features[f'{mod}_skewness'],
            mri_features[f'{mod}_kurtosis'],
            mri_features[f'{mod}_glcm_contrast_mean'],
            mri_features[f'{mod}_glcm_contrast_std'],
            mri_features[f'{mod}_gradient_magnitude_mean']
        ])
    
    # Add cross-modality features
    feature_vector.append(mri_features['combined_center_of_mass'][0])  # x
    feature_vector.append(mri_features['combined_center_of_mass'][1])  # y
    feature_vector.append(mri_features['combined_center_of_mass'][2])  # z
    feature_vector.append(mri_features['combined_entropy'])  # entropy
    
    # Append to the CSV file
    new_row = pd.DataFrame([feature_vector], columns=headers)
    new_row.to_csv(results_save_path, mode='a', header=False, index=False)

    # Cleanup to release GPU memory
    del image, batch_data, mri_features, feature_vector
    gc.collect()

print(f"Processing complete. All MRI features saved to {results_save_path}.")


