import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from radiomics import featureextractor # Requires usage pf myenc environment

# --- Configuration ---
json_path = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge\splits\data_splits.json"
base_dir = r"\\wsl.localhost\Ubuntu-22.04\home\magata\data\brats2021challenge"  
output_csv = "test_set_tumor_stats_all_modalities.csv"
test_key = "test"
 
modalities = ["flair", "t1ce", "t1", "t2"]

# Selected radiomics features to extract
selected_rad_features = [
    "original_firstorder_Entropy",
    "original_glcm_Contrast",
    "original_glcm_Homogeneity",
    "original_gldm_DependenceEntropy"
]

# --- Initialize the Pyradiomics feature extractor ---
extractor = featureextractor.RadiomicsFeatureExtractor()

# --- Load JSON file with data splits ---
with open(json_path, "r") as f:
    data_splits = json.load(f)

if test_key in data_splits:
    patients = data_splits[test_key]
else:
    print(f"No '{test_key}' key found in JSON. Falling back to 'training' set.")
    patients = data_splits.get("training", [])

results = []

for patient in patients:
    # Extract patient ID from the segmentation label path.
    label_rel_path = patient["label"]
    try:
        base_name = os.path.basename(label_rel_path)
        patient_id = base_name.split('_')[1]  # e.g., "01551"
    except Exception as e:
        print(f"Could not extract patient_id from {label_rel_path}: {e}")
        continue

    # Construct full path for the segmentation (ground truth) file.
    label_full_path = os.path.join(base_dir, label_rel_path)
    try:
        seg_img = nib.load(label_full_path)
    except Exception as e:
        print(f"Error loading segmentation for patient {patient_id}: {e}")
        continue

    seg_data = seg_img.get_fdata().astype(np.int32)
    # Create a tumor mask (assuming any nonzero value indicates tumor).
    tumor_mask = seg_data > 0

    # Compute voxel volume from the segmentation affine (assumed co-registered)
    affine = seg_img.affine
    voxel_volume = np.abs(np.linalg.det(affine[:3, :3]))  # in mm^3
    tumor_volume = np.sum(tumor_mask) * voxel_volume  # volume in mm^3

    # Create an output dictionary for the patient.
    patient_features = {
        "patient_id": patient_id,
        "tumor_volume_mm3": tumor_volume
    }

    # For each modality, find the corresponding image (if available) and extract features.
    for mod in modalities:
        mod_path = None
        for img_rel_path in patient["image"]:
            if mod.lower() in img_rel_path.lower():
                mod_path = img_rel_path
                break
        if mod_path is None:
            # If a modality is missing, record NaN for its features.
            patient_features[f"{mod}_intensity_mean"] = np.nan
            patient_features[f"{mod}_intensity_std"]  = np.nan
            patient_features[f"{mod}_intensity_min"]  = np.nan
            patient_features[f"{mod}_intensity_max"]  = np.nan
            # Also set radiomics features as NaN.
            for feat in selected_rad_features:
                patient_features[f"{mod}_radiomics_{feat}"] = np.nan
            continue

        # Construct the full path for the modality image.
        mod_full_path = os.path.join(base_dir, mod_path)
        try:
            mod_img = nib.load(mod_full_path)
        except Exception as e:
            print(f"Error loading {mod} image for patient {patient_id}: {e}")
            patient_features[f"{mod}_intensity_mean"] = np.nan
            patient_features[f"{mod}_intensity_std"]  = np.nan
            patient_features[f"{mod}_intensity_min"]  = np.nan
            patient_features[f"{mod}_intensity_max"]  = np.nan
            for feat in selected_rad_features:
                patient_features[f"{mod}_radiomics_{feat}"] = np.nan
            continue

        mod_data = mod_img.get_fdata()
        if mod_data.shape != seg_data.shape:
            print(f"Shape mismatch for patient {patient_id} modality {mod}: {mod_data.shape} vs {seg_data.shape}")
        
        # Intensity statistics within the tumor region.
        tumor_intensities = mod_data[tumor_mask]
        if tumor_intensities.size == 0:
            patient_features[f"{mod}_intensity_mean"] = np.nan
            patient_features[f"{mod}_intensity_std"]  = np.nan
            patient_features[f"{mod}_intensity_min"]  = np.nan
            patient_features[f"{mod}_intensity_max"]  = np.nan
        else:
            patient_features[f"{mod}_intensity_mean"] = float(np.mean(tumor_intensities))
            patient_features[f"{mod}_intensity_std"]  = float(np.std(tumor_intensities))
            patient_features[f"{mod}_intensity_min"]  = float(np.min(tumor_intensities))
            patient_features[f"{mod}_intensity_max"]  = float(np.max(tumor_intensities))
        
        # Save a temporary mask image for the modality.
        # Convert boolean tumor_mask to uint8 (1 = tumor, 0 = background).
        mask_temp_path = f"temp_mask_{patient_id}_{mod}.nii.gz"
        temp_mask_img = nib.Nifti1Image(tumor_mask.astype(np.uint8), mod_img.affine)
        nib.save(temp_mask_img, mask_temp_path)

        # Use Pyradiomics to extract radiomics features.
        try:
            rad_features = extractor.execute(mod_full_path, mask_temp_path)
        except Exception as e:
            print(f"Radiomics extraction failed for patient {patient_id}, modality {mod}: {e}")
            rad_features = {}

        # Save selected radiomics features.
        for feat in selected_rad_features:
            key = f"{mod}_radiomics_{feat}"
            # Radiomics results often have keys prefixed with "original_".
            if feat in rad_features:
                patient_features[key] = rad_features[feat]
            else:
                patient_features[key] = np.nan

        # Remove the temporary mask file.
        try:
            os.remove(mask_temp_path)
        except Exception as e:
            print(f"Could not remove temporary mask {mask_temp_path}: {e}")

    results.append(patient_features)

# Create a DataFrame and save to CSV.
df = pd.DataFrame(results)
print(df)
df.to_csv(output_csv, index=False)
print(f"Extracted features for {len(results)} patients and saved to {output_csv}")
