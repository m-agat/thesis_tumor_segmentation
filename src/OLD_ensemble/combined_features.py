import pandas as pd

region = "ET"
# Load the previously saved selected features for each model
swinunetr_features_df = pd.read_csv(f"./outputs/final_features_swinunetr_{region}_meta_learner.csv")  # Replace with the correct file path
segresnet_features_df = pd.read_csv(f"./outputs/final_features_segresnet_{region}_meta_learner.csv")  # Replace with the correct file path

# Extract the list of features from the columns in the dataframes (assuming features are stored in a column)
swinunetr_features = swinunetr_features_df['Features'].tolist()
segresnet_features = segresnet_features_df['Features'].tolist()

# Step 1: Create the union of features (all unique features from both models)
combined_features = list(set(swinunetr_features + segresnet_features))

# Step 2: Save the combined features into a new CSV file
combined_features_df = pd.DataFrame(combined_features, columns=["Feature"])
combined_features_df.to_csv(f"./outputs/combined_selected_features_{region}.csv", index=False)

print(f"Combined selected features saved to './outputs/combined_selected_features_{region}.csv'.")
