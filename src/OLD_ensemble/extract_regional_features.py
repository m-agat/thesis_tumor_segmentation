import os
import sys
import gc
import numpy as np
import pandas as pd
import psutil

# Local modules
sys.path.append("../")
import config.config as config
from mri_feature_extraction import extract_features_from_tensor, extract_regional_features

total_patients = len(config.val_loader)

# Prepare output path
results_save_path = os.path.join(config.output_dir, "mri_regional_features.csv")

# Initialize the CSV with headers
headers = ['Patient']
for mod in ["flair", "t1", "t1ce", "t2"]:
    for region in ["NCR", "ED", "ET"]:
        headers.extend([
            f'{mod}_{region}_mean_intensity', f'{mod}_{region}_stddev_intensity',
            f'{mod}_{region}_entropy', f'{mod}_{region}_sobel_mean', f'{mod}_{region}_sobel_std',
            f'{mod}_{region}_gabor_mean_freq_0.1', f'{mod}_{region}_gabor_std_freq_0.1',
            f'{mod}_{region}_gabor_mean_freq_0.5', f'{mod}_{region}_gabor_std_freq_0.5',
            f'{mod}_{region}_gabor_mean_freq_0.8', f'{mod}_{region}_gabor_std_freq_0.8',
            f'{mod}_{region}_lbp_mean', f'{mod}_{region}_lbp_std',
            f'{mod}_{region}_skewness', f'{mod}_{region}_kurtosis',
            f'{mod}_{region}_glcm_contrast_mean', f'{mod}_{region}_glcm_contrast_std',
            f'{mod}_{region}_gradient_magnitude_mean'
        ])

# Save headers to the CSV
df = pd.DataFrame(columns=headers)
df.to_csv(results_save_path, index=False)

# Process each patient and save features
total_patients = len(config.val_loader)
headers_initialized = False

for idx, batch_data in enumerate(config.val_loader):
    image = batch_data["image"][0]
    gt = batch_data["label"][0].cpu().numpy()  
    patient_path = batch_data["path"]
    patient_id = os.path.basename(patient_path[0])  

    ncr, ed, et = 1, 2, 4
    masks = {
        "NCR": (gt == ncr),
        "ED": (gt == ed),
        "ET": (gt == et)
    }
    
    print(f"Processing patient {idx + 1}/{total_patients}: {patient_id}")

    # Extract regional features
    regional_features = extract_regional_features(image, masks)

    if not headers_initialized:
        headers = ['Patient'] + list(regional_features.keys())
        df = pd.DataFrame(columns=headers)
        df.to_csv(results_save_path, index=False, mode='w')  # Save headers to CSV
        headers_initialized = True

    # Build feature vector in the order of headers
    feature_vector = [patient_id]  # Include patient ID
    for key in headers[1:]:  # Skip the 'Patient' header
        value = regional_features.get(key, np.nan)  # Use `get` to handle missing keys
        feature_vector.append(value)

    print("Feature vector to save:", feature_vector)

    if len(feature_vector) != len(headers):
        print(f"Error: Mismatch between headers and feature vector lengths! ({len(headers)} vs {len(feature_vector)})")
    else:
        # Append to the CSV file
        new_row = pd.DataFrame([feature_vector], columns=headers)
        new_row.to_csv(results_save_path, mode='a', header=False, index=False)

    # Cleanup to release GPU memory
    del image, batch_data, feature_vector
    gc.collect()

print(f"Processing complete. All MRI features saved to {results_save_path}.")


