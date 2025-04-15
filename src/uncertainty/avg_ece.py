import pandas as pd

df = pd.read_csv("../ensemble/output_segmentations/ttd/ece_results_per_subregion.csv")

# Compute the average ECE for each subregion (NCR, ED, ET)
# Calculate mean for each subregion column
avg_ece_subregions = {
    'Overall': df['Overall_ECE'].mean(),
    'NCR': df['NCR_ECE'].mean(), 
    'ED': df['ED_ECE'].mean(),
    'ET': df['ET_ECE'].mean()
}


print(avg_ece_subregions)
