import pandas as pd 

# Load and prepare model results
model_results = {
    "AttentionUNet": pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/average_dice_scores.csv").T,
    "SegResNet": pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/average_dice_scores.csv").T,
    "SwinUNetr": pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/average_dice_scores.csv").T,
    "VNet": pd.read_csv("/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/average_dice_scores.csv").T
}

# Prepare and clean up each model's DataFrame
for model in model_results.keys():
    new_header = model_results[model].iloc[0] 
    df = model_results[model][1:] 
    df.columns = new_header
    model_results[model] = df

# Concatenate into a single DataFrame
all_model_scores = pd.concat(model_results, names=["Model", "Index"]).reset_index()
all_model_scores.columns.name = None  # Remove extra column name
all_model_scores.drop(columns=["Index"], inplace=True)  # Drop "Index" column

# Save to CSV
output_path = "/home/agata/Desktop/thesis_tumor_segmentation/results/model_weights.csv"
all_model_scores.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")