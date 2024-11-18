import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Combine unique top features
combined_features_df = pd.read_csv("./outputs/combined_selected_features.csv")
selected_features = combined_features_df['Feature'].tolist()
print(selected_features)

def convert_tuple_to_mean(value):
    if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
        try:
            return np.mean(eval(value))
        except:
            return np.nan
    return value

# Load data
features_df = pd.read_csv("./outputs/mri_regional_features.csv")
performance_df_swinunetr = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_swinunetr.csv")
performance_df_segnet = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_segresnet.csv")

# # Merge features with performance metrics
merged_swinunetr = pd.merge(features_df, performance_df_swinunetr, on='Patient')
merged_segnet = pd.merge(features_df, performance_df_segnet, on='Patient')

# Select features
features = features_df[selected_features].copy()
for col in features.columns:
    features[col] = features[col].apply(convert_tuple_to_mean)
features = features.fillna(features.mean())  # Handle missing values

# Normalize composite scores to create pseudo-weights
pseudo_weights = pd.DataFrame({
    'SwinUNETR_Weight': performance_df_swinunetr['Composite_Score_1'] / (
        performance_df_swinunetr['Composite_Score_1'] + performance_df_segnet['Composite_Score_1'] + 1e-8
    ),
    'SegResNet_Weight': performance_df_segnet['Composite_Score_1'] / (
        performance_df_swinunetr['Composite_Score_1'] + performance_df_segnet['Composite_Score_1'] + 1e-8
    )
})
valid_indices = pseudo_weights.dropna().index
pseudo_weights = pseudo_weights.loc[valid_indices]
features = features.loc[valid_indices]

# Validate no NaN values remain
assert not pseudo_weights.isnull().values.any(), "Pseudo-weights contain NaN values!"
assert not features.isnull().values.any(), "Features contain NaN values!"

# Prepare targets
target = pseudo_weights

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train meta-learner (Random Forest)
meta_learner = RandomForestRegressor(random_state=42, n_estimators=100)
meta_learner.fit(X_train, y_train)

# Predict weights
y_pred = meta_learner.predict(X_test)

# Evaluate the meta-learner
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (Pseudo-Weights Prediction): {mse}")
print(f"R2 Score (Pseudo-Weights Prediction): {r2}")

# Compare predicted vs actual pseudo-weights
comparison = pd.DataFrame({
    'Actual_SwinUNETR_Weight': y_test['SwinUNETR_Weight'].values,
    'Predicted_SwinUNETR_Weight': y_pred[:, 0],
    'Actual_SegResNet_Weight': y_test['SegResNet_Weight'].values,
    'Predicted_SegResNet_Weight': y_pred[:, 1],
})
comparison.to_csv("./outputs/swinunetr_segresnet_preds.csv")
print("Comparison of Actual vs Predicted Weights (First 5 Examples):")
print(comparison.head())

#Visualize weight distributions
import matplotlib.pyplot as plt

plt.hist(y_test['SwinUNETR_Weight'], alpha=0.5, label='Actual SwinUNETR Weights')
plt.hist(y_pred[:, 0], alpha=0.5, label='Predicted SwinUNETR Weights')
plt.legend()
plt.title("Weight Distribution: SwinUNETR")
plt.savefig("./outputs/weight_distr_swinunet.png")

plt.hist(y_test['SegResNet_Weight'], alpha=0.5, label='Actual SegResNet Weights')
plt.hist(y_pred[:, 1], alpha=0.5, label='Predicted SegResNet Weights')
plt.legend()
plt.title("Weight Distribution: SegResNet")
plt.savefig("./outputs/weight_distr_segnet.png")
