import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import shap

def convert_tuple_to_mean(value):
    if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
        try:
            return np.mean(eval(value))
        except:
            return np.nan
    return value

# Load data
model = "swinunetr"
region = "NCR"
target_col_name = 'Composite_Score_1'
features_df = pd.read_csv("./outputs/mri_regional_features.csv")
performance_df = pd.read_csv(f"/home/magata/results/metrics/patient_performance_scores_{model}.csv")

# Merge features with performance metrics
merged_df = pd.merge(features_df, performance_df, on='Patient')
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Select only MRI feature columns (exclude Patient and performance metrics)
performance_columns = [col for col in performance_df.columns if col not in ['Patient', target_col_name]]
columns_to_exclude = ['Patient'] + performance_columns

region_columns = [col for col in features_df.columns if region in col]
features = merged_df[region_columns]
for col in features.columns:
    features[col] = features[col].apply(convert_tuple_to_mean)
features = features.select_dtypes(include=[np.number])

target = merged_df[target_col_name]  

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Fine-tune the XGBoost using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb.XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_xgb_model = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Step 4: Get feature importances from the trained model (after fitting)
importances = best_xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Step 5: Combine the filtered features based on importance
selected_importance_features = importance_df.head(10)['Feature'].tolist()
print(f"Top 10 features based on feature importance: {selected_importance_features}")

# Step 7: Save the final selected features
final_features_df = pd.DataFrame(selected_importance_features, columns=["Features"])
final_features_df.to_csv(f"./outputs/final_features_{model}_{region}_meta_learner.csv", index=False)
print(f"Final selected features saved to './outputs/final_features_{model}_{region}_meta_learner.csv'.")

# Train the meta-learner (XGBoost)
meta_learner = xgb.XGBRegressor(random_state=42, n_estimators=100)
meta_learner.fit(X_train[selected_importance_features], y_train)

# Predict and evaluate the meta-learner
y_pred = meta_learner.predict(X_test[selected_importance_features])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (Test Set): {mse}")
print(f"R2 Score (Test Set): {r2}")

# SHAP analysis for feature contribution
explainer = shap.TreeExplainer(meta_learner)
shap_values = explainer.shap_values(X_test[selected_importance_features])

# # Plot SHAP summary
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test[selected_importance_features], feature_names=selected_importance_features, show=False)
plt.savefig(f"./outputs/shap_summary_plot_{model}_{region}.png", dpi=300, bbox_inches="tight")
print(f"SHAP summary plot saved as 'shap_summary_{model}plot.png' in './outputs/'.")
