import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle as pkl

# Combine unique top features
region = "NCR"
performance_metric = "Composite_Score_1"
combined_features_df = pd.read_csv(f"./outputs/combined_selected_features_{region}.csv")
selected_features = combined_features_df['Feature'].tolist()

def convert_tuple_to_mean(value):
    if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
        try:
            return np.mean(eval(value))
        except:
            return np.nan
    return value

# Load data
features_df = pd.read_csv("./outputs/mri_features.csv")
performance_df_swinunetr = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_swinunetr.csv")
performance_df_segnet = pd.read_csv("/home/magata/results/metrics/patient_performance_scores_segresnet.csv")

# Select features
# features = features_df[selected_features].copy()
features = features_df.copy()
for col in features.columns:
    features[col] = features[col].apply(convert_tuple_to_mean)
features = features.select_dtypes(include=[np.number])
features = features.fillna(features.mean())

# Normalize composite scores to create pseudo-weights
pseudo_weights = pd.DataFrame({
    'SwinUNETR_Weight': performance_df_swinunetr[performance_metric] / (
        performance_df_swinunetr[performance_metric] + performance_df_segnet[performance_metric] + 1e-8
    ),
    'SegResNet_Weight': performance_df_segnet[performance_metric] / (
        performance_df_swinunetr[performance_metric] + performance_df_segnet[performance_metric] + 1e-8
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

# model_filename = f"./outputs/meta_learner_{region}.pkl"
# with open(model_filename, 'wb') as f:
    # pkl.dump(meta_learner, f)
# print(f"Model saved to {model_filename}")

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
