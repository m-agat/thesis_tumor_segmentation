import pandas as pd

# Load CSV files for different models
swinunetr_scores = pd.read_csv('/home/agata/Desktop/thesis_tumor_segmentation/results/SwinUNetr/patient_dice_scores.csv')  
segresnet_scores = pd.read_csv('/home/agata/Desktop/thesis_tumor_segmentation/results/SegResNet/patient_dice_scores.csv')
attunet_scores = pd.read_csv('/home/agata/Desktop/thesis_tumor_segmentation/results/AttentionUNet/patient_dice_scores.csv')
# model_4_scores = pd.read_csv('model_4_dice_scores.csv')
# 
# Merge data on the 'Patient' column
merged_scores = swinunetr_scores.merge(segresnet_scores, on='Patient', suffixes=('_m1', '_m2'))
merged_scores = merged_scores.merge(attunet_scores, on='Patient', suffixes=('', '_m3'))
print(merged_scores)
# merged_scores = merged_scores.merge(model_4_scores, on='Patient', suffixes=('', '_m4'))

# Define a function to detect inconsistency for a specific class
def is_inconsistent(dice_scores, threshold=0.2):
    """
    Check if the performance is inconsistent across models.
    
    Args:
        dice_scores: List of Dice scores from all models for a specific tumor class.
        threshold: Difference threshold to consider as inconsistent performance.
        
    Returns:
        True if the difference between the highest and lowest score is greater than the threshold, False otherwise.
    """
    return (max(dice_scores) - min(dice_scores)) >= threshold

# Iterate over each class (Dice_1, Dice_2, Dice_4) and check for inconsistencies
inconsistent_cases = {'Dice_1': [], 'Dice_2': [], 'Dice_4': []}
for index, row in merged_scores.iterrows():
    # Check Dice_1 scores
    dice_1_scores = [row['Dice_1_m1'], row['Dice_1_m2'], row['Dice_1']]
    if is_inconsistent(dice_1_scores):
        inconsistent_cases['Dice_1'].append(row['Patient'])
    
    # Check Dice_2 scores
    dice_2_scores = [row['Dice_2_m1'], row['Dice_2_m2'], row['Dice_2']]
    if is_inconsistent(dice_2_scores):
        inconsistent_cases['Dice_2'].append(row['Patient'])
    
    # Check Dice_4 scores
    dice_4_scores = [row['Dice_4_m1'], row['Dice_4_m2'], row['Dice_4']]
    if is_inconsistent(dice_4_scores):
        inconsistent_cases['Dice_4'].append(row['Patient'])

# Output inconsistent cases
print("Inconsistent Cases for Dice_1 (NCR):", inconsistent_cases['Dice_1'])
print("Inconsistent Cases for Dice_2 (ED):", inconsistent_cases['Dice_2'])
print("Inconsistent Cases for Dice_4 (ET):", inconsistent_cases['Dice_4'])

# Optional: Save inconsistent cases to CSV files
pd.DataFrame({'Patient': inconsistent_cases['Dice_1']}).to_csv('inconsistent_dice_1.csv', index=False)
pd.DataFrame({'Patient': inconsistent_cases['Dice_2']}).to_csv('inconsistent_dice_2.csv', index=False)
pd.DataFrame({'Patient': inconsistent_cases['Dice_4']}).to_csv('inconsistent_dice_4.csv', index=False)
