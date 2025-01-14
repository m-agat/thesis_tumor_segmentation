import pandas as pd

brats_mapping = "/home/magata/data/metadata/BraTS2021_MappingToTCIA.xlsx"
ucsf_pdgm_path = "/home/magata/data/metadata/UCSF-PDGM-metadata_v2.csv"
cptac_gbm_path = "/home/magata/data/metadata/CPTAC_GBM_mmc2.xlsx"

brats_mapping_df = pd.read_excel(brats_mapping)
ucsf_pdgm_df = pd.read_csv(ucsf_pdgm_path)

brats_ucsf_pdgm = brats_mapping_df[brats_mapping_df['Data Collection (as on TCIA+additional)'] == 'UCSF-PDGM']

# Filter UCSF-PDGM patients only in the "Training" cohort
brats_ucsf_pdgm_training = brats_ucsf_pdgm[brats_ucsf_pdgm['Segmentation (Task 1) Cohort'] == 'Training']
brats_training_ids = set(brats_ucsf_pdgm_training['BraTS2021 ID'])

# Merge the two DataFrames based on BraTS2021 ID for the training cohort
merged_training_df = pd.merge(brats_ucsf_pdgm_training, ucsf_pdgm_df, left_on='BraTS2021 ID', right_on='BraTS21 ID', how='inner')

# Analyze tumor types in the training cohort
glioblastoma_count = merged_training_df[merged_training_df['Final pathologic diagnosis (WHO 2021)'].str.contains('Glioblastoma', case=False, na=False)].shape[0]
astrocytoma_count = merged_training_df[merged_training_df['Final pathologic diagnosis (WHO 2021)'].str.contains('Astrocytoma', case=False, na=False)].shape[0]

print(f"Number of training cohort patients with glioblastoma: {glioblastoma_count}")
print(f"Number of training cohort patients with astrocytoma: {astrocytoma_count}")

# Group by diagnosis and WHO CNS Grade, and count
tumor_grade_distribution = merged_training_df.groupby(['Final pathologic diagnosis (WHO 2021)', 'WHO CNS Grade']).size().reset_index(name='Count')

# Display in desired format
print("Tumor Grade Distribution:")
for _, row in tumor_grade_distribution.iterrows():
    print(f"{row['Final pathologic diagnosis (WHO 2021)']} | Grade {row['WHO CNS Grade']} | {row['Count']}")