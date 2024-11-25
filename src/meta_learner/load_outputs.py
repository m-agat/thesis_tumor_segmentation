import os
import numpy as np
import nibabel as nib
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader


def save_file_paths(base_path, models, ground_truth_dir):
    data = []
    for model_name in models:
        model_output_dir = os.path.join(base_path, model_name)
        segmentation_files = [f for f in os.listdir(model_output_dir) if "_segmentation.nii.gz" in f]
        probability_files = [f for f in os.listdir(model_output_dir) if "_probability.nii.gz" in f]
        probability_files = [f for f in os.listdir(model_output_dir) if "_logits.nii.gz" in f]

        for seg_file, prob_file in zip(segmentation_files, probability_files):
            patient_id = seg_file.split('_')[0] + '_' + seg_file.split('_')[1]

            gt_file = f"{patient_id}_seg.nii.gz"
            gt_path = os.path.join(os.path.join(ground_truth_dir, patient_id), gt_file)

            if not os.path.exists(gt_path):
                print(f"Ground truth file not found for patient {patient_id}, skipping.")
                continue  

            patient_exists = next((item for item in data if item["Patient"] == patient_id), None)

            if patient_exists:
                patient_exists[f"{model_name}_segmentation"] = os.path.join(model_output_dir, seg_file)
                patient_exists[f"{model_name}_probability"] = os.path.join(model_output_dir, prob_file)
                patient_exists[f"{model_name}_logits"] = os.path.join(model_output_dir, prob_file)
            else:
                data.append({
                    "Patient": patient_id,
                    "Ground_Truth": gt_path,
                    f"{model_name}_segmentation": os.path.join(model_output_dir, seg_file),
                    f"{model_name}_probability": os.path.join(model_output_dir, prob_file),
                    f"{model_name}_logits": os.path.join(model_output_dir, prob_file)
                })
    
    data_df = pd.DataFrame(data)
    data_df.to_csv("./outputs/model_file_paths_with_gt.csv", index=False)
    print("File paths including ground truth saved to model_file_paths_with_gt.csv")

class MetaLearnerDataset(Dataset):
    def __init__(self, data_df, models, patch_size=(64, 64, 64), stride=(32, 32, 32)):
        self.data_df = data_df
        self.models = models
        self.patch_size = patch_size
        self.stride = stride
        self.data = []

        # Precompute all patches and store their coordinates
        for idx in range(len(self.data_df)):
            self.data.extend(self._generate_patches(idx))

    def _generate_patches(self, idx):
        row = self.data_df.iloc[idx]
        patient_input = []

        for model_name in self.models:
            logits_path = row[f"{model_name}_logits"]
            logits_nifti = nib.load(logits_path)
            logits_map = logits_nifti.get_fdata()
            patient_input.append(logits_map)

        # Stack all model outputs along the channel dimension
        patient_input = np.stack(patient_input, axis=0)

        # Ground truth
        ground_truth_path = row["Ground_Truth"]
        ground_truth_nifti = nib.load(ground_truth_path)
        ground_truth_mask = ground_truth_nifti.get_fdata()

        # Extract patches and store them
        input_patches = self.create_patches(patient_input)
        gt_patches = self.create_patches(ground_truth_mask, is_mask=True)

        return [(input_patches[i], gt_patches[i]) for i in range(len(input_patches))]

    def create_patches(self, volume, is_mask=False):
        if is_mask:
            volume = np.expand_dims(volume, axis=(0, 1))

        M, C, H, W, D = volume.shape
        patch_size = self.patch_size
        stride = self.stride

        num_patches_H = (H - patch_size[0]) // stride[0] + 1
        num_patches_W = (W - patch_size[1]) // stride[1] + 1
        num_patches_D = (D - patch_size[2]) // stride[2] + 1

        patches = []
        for h in range(0, num_patches_H * stride[0], stride[0]):
            for w in range(0, num_patches_W * stride[1], stride[1]):
                for d in range(0, num_patches_D * stride[2], stride[2]):
                    patch = volume[:, :, h:h+patch_size[0], w:w+patch_size[1], d:d+patch_size[2]]
                    patches.append(patch)

        return np.stack(patches)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_patch, gt_patch = self.data[idx]
        return {
            "image": torch.tensor(input_patch, dtype=torch.float32),
            "label": torch.tensor(gt_patch, dtype=torch.long),
        }


# base_path = "./outputs/"  # The path where the model outputs (segmentation, probability maps, and logits) are stored
# models = ["swinunetr", "segresnet", "vnet", "attunet"]  
# gt_path = "/home/magata/data/brats2021challenge/split/val/" 
# save_file_paths(base_path, models, gt_path) 

# data_df = pd.read_csv("./outputs/model_file_paths_with_gt.csv")  # Load your CSV with paths

# dataset = MetaLearnerDataset(data_df, models)

# batch_size = 1
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # Iterate over the DataLoader
# for X_batch, y_batch in dataloader:
#     print(X_batch.shape, y_batch.shape) 
#     break
