import pandas as pd

# Load CSV files for different models
swinunetr_scores = pd.read_csv('/home/magata/results/metrics/patient_performance_scores_swinunetr.csv')  
segresnet_scores = pd.read_csv('/home/magata/results/metrics/patient_performance_scores_segresnet.csv')
attunet_scores = pd.read_csv('/home/magata/results/metrics/patient_performance_scores_attunet.csv')
vnet_scores = pd.read_csv('/home/magata/results/metrics/patient_performance_scores_vnet.csv')

swinunetr_scores = swinunetr_scores.rename(columns=lambda col: col if col == 'Patient' else f'{col}_m1')
segresnet_scores = segresnet_scores.rename(columns=lambda col: col if col == 'Patient' else f'{col}_m2')
attunet_scores = attunet_scores.rename(columns=lambda col: col if col == 'Patient' else f'{col}_m3')
vnet_scores = vnet_scores.rename(columns=lambda col: col if col == 'Patient' else f'{col}_m4')

# Merge data on the 'Patient' column
merged_scores = swinunetr_scores.merge(segresnet_scores, on='Patient')
merged_scores = merged_scores.merge(attunet_scores, on='Patient')
merged_scores = merged_scores.merge(vnet_scores, on='Patient')
# print(merged_scores.info())  
# print(merged_scores.head())

# Define a function to detect inconsistency for a specific class
def is_inconsistent(scores, threshold=0.2):
    """
    Check if the performance is inconsistent across models.
    
    Args:
        scores: List of scores from all models for a specific tumor class.
        threshold: Difference threshold to consider as inconsistent performance.
        
    Returns:
        True if the difference between the highest and lowest score is greater than the threshold, False otherwise.
    """
    return (max(scores) - min(scores)) >= threshold

# # Iterate over each class (Dice_1, Dice_2, Dice_4) and check for inconsistencies
inconsistent_cases = {'Dice_1': [], 'Dice_2': [], 'Dice_4': []}

for index, row in merged_scores.iterrows():
    # Check Dice_1 scores (NCR)
    dice_1_scores = [row['Dice_1_m1'], row['Dice_1_m2'], row['Dice_1_m3'], row['Dice_1_m4']]
    if is_inconsistent(dice_1_scores):
        inconsistent_cases['Dice_1'].append({
            'Patient': row['Patient'],
            'SwinUNETR': row['Dice_1_m1'],
            'SegResNet': row['Dice_1_m2'],
            'AttUNET': row['Dice_1_m3'],
            'VNet': row['Dice_1_m4']
        })
    
    # Check Dice_2 scores (ED)
    dice_2_scores = [row['Dice_2_m1'], row['Dice_2_m2'], row['Dice_2_m3'], row['Dice_2_m4']]
    if is_inconsistent(dice_2_scores):
        inconsistent_cases['Dice_2'].append({
            'Patient': row['Patient'],
            'SwinUNETR': row['Dice_2_m1'],
            'SegResNet': row['Dice_2_m2'],
            'AttUNET': row['Dice_2_m3'],
            'VNet': row['Dice_2_m4']
        })
    
    # Check Dice_4 scores (ET)
    dice_4_scores = [row['Dice_4_m1'], row['Dice_4_m2'], row['Dice_4_m3'], row['Dice_4_m4']]
    if is_inconsistent(dice_4_scores):
        inconsistent_cases['Dice_4'].append({
            'Patient': row['Patient'],
            'SwinUNETR': row['Dice_4_m1'],
            'SegResNet': row['Dice_4_m2'],
            'AttUNET': row['Dice_4_m3'],
            'VNet': row['Dice_4_m4']
        })

# Save inconsistent cases to CSV files for each class
for dice_class, cases in inconsistent_cases.items():
    if cases:
        inconsistent_df = pd.DataFrame(cases)
        file_name = f'inconsistent_{dice_class.lower()}.csv'
        inconsistent_df.to_csv(file_name, index=False)
        print(f"Inconsistent cases for {dice_class} saved to {file_name}.")