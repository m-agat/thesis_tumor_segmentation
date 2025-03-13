import pandas as pd

df_swinunetr = pd.read_csv(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/average_dice_scores.csv"
)
df_segresnet = pd.read_csv(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/average_dice_scores.csv"
)
df_attunet = pd.read_csv(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/average_dice_scores.csv"
)
df_vnet = pd.read_csv(
    "/home/agata/Desktop/thesis_tumor_segmentation/results/VNet/average_dice_scores.csv"
)

# Assuming these are your existing dataframes
df_swinunetr.rename(columns={"Average Dice": "SwinUNETR"}, inplace=True)
df_segresnet.rename(columns={"Average Dice": "SegResNet"}, inplace=True)
df_attunet.rename(columns={"Average Dice": "AttentionUNet"}, inplace=True)
df_vnet.rename(columns={"Average Dice": "VNet"}, inplace=True)

# Convert Tissue column to labels (1, 2, 4)
tissue_to_label = {"NCR (1)": 1, "ED (2)": 2, "ET (4)": 4}
df_swinunetr["Tissue"] = df_swinunetr["Tissue"].map(tissue_to_label)
df_segresnet["Tissue"] = df_segresnet["Tissue"].map(tissue_to_label)
df_attunet["Tissue"] = df_attunet["Tissue"].map(tissue_to_label)
df_vnet["Tissue"] = df_vnet["Tissue"].map(tissue_to_label)

# Merge dataframes on the "Tissue" column
merged_df = (
    df_swinunetr.merge(df_segresnet, on="Tissue")
    .merge(df_attunet, on="Tissue")
    .merge(df_vnet, on="Tissue")
)

print(merged_df)
