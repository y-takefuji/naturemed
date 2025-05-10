import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import spearmanr, kendalltau
from minepy import MINE

# 1. Load data - Full feature set (12 features)
df = pd.read_csv('cardio_train.csv', sep=';')
X = df.drop('cardio', axis=1)
y = df['cardio']
feat_names = X.columns.tolist()
print(f"Full dataset - Instances: {X.shape[0]}, Features: {X.shape[1]}")
print(f"Full feature set ({len(feat_names)} features): {feat_names}")

# 2. Feature selection methods to find top features
def select_top_features(data, target, method_name, n=10):  # Changed default from 11 to 10
    importance = {}
    
    if method_name == 'xgboost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(data, target)
        importance = dict(zip(data.columns, model.feature_importances_))
    elif method_name == 'spearman':
        for col in data.columns:
            corr, _ = spearmanr(data[col], target)
            importance[col] = abs(corr)
    elif method_name == 'kendall':
        for col in data.columns:
            corr, _ = kendalltau(data[col], target)
            importance[col] = abs(corr)
    elif method_name == 'mic':
        mine = MINE(alpha=0.6, c=15)
        for col in data.columns:
            mine.compute_score(data[col], target)
            importance[col] = mine.mic()

    # Get top n features sorted by importance
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n]
    return [feat for feat, _ in top_features], importance

# 3. Top 10 feature selection from full feature set (12 features)
xgb_top10, xgb_importance = select_top_features(X, y, method_name='xgboost', n=10)
spearman_top10, spearman_importance = select_top_features(X, y, method_name='spearman', n=10)
kendall_top10, kendall_importance = select_top_features(X, y, method_name='kendall', n=10)
mic_top10, mic_importance = select_top_features(X, y, method_name='mic', n=10)

# 4. Create reduced datasets with top 10 features from each method
X_xgb_reduced = X[xgb_top10]
X_spearman_reduced = X[spearman_top10]
X_kendall_reduced = X[kendall_top10]
X_mic_reduced = X[mic_top10]

# 5. Print the top 10 features selected by each method
print("\n========== TOP 10 FEATURES SELECTED FROM FULL DATASET ==========")
print(f"XGBoost:  {xgb_top10}")
print(f"Spearman: {spearman_top10}")
print(f"Kendall:  {kendall_top10}")
print(f"MIC:      {mic_top10}")

# 6. Extract top 8 features from full feature set
xgb_top8_full, _ = select_top_features(X, y, method_name='xgboost', n=8)
spearman_top8_full, _ = select_top_features(X, y, method_name='spearman', n=8)
kendall_top8_full, _ = select_top_features(X, y, method_name='kendall', n=8)
mic_top8_full, _ = select_top_features(X, y, method_name='mic', n=8)

# 7. Extract top 8 features from reduced feature sets
xgb_top8_reduced, _ = select_top_features(X_xgb_reduced, y, method_name='xgboost', n=8)
spearman_top8_reduced, _ = select_top_features(X_spearman_reduced, y, method_name='spearman', n=8)
kendall_top8_reduced, _ = select_top_features(X_kendall_reduced, y, method_name='kendall', n=8)
mic_top8_reduced, _ = select_top_features(X_mic_reduced, y, method_name='mic', n=8)

# 8. Print top 8 features from both full and reduced datasets
print("\n========== TOP 8 FEATURES ==========")
print("From full dataset (12 features):")
print(f"XGBoost:  {xgb_top8_full}")
print(f"Spearman: {spearman_top8_full}")
print(f"Kendall:  {kendall_top8_full}")
print(f"MIC:      {mic_top8_full}")

print("\nFrom reduced dataset (10 features):")  # Updated comment
print(f"XGBoost:  {xgb_top8_reduced}")
print(f"Spearman: {spearman_top8_reduced}")
print(f"Kendall:  {kendall_top8_reduced}")
print(f"MIC:      {mic_top8_reduced}")

# 9. Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model_cv = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# 9a. Cross-validation with full feature set (12 features)
print("\n========== CROSS-VALIDATION WITH FULL FEATURE SET (12 FEATURES) ==========")
full_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='accuracy')
print(f"Full dataset accuracy: {full_scores.mean():.3f}±{full_scores.std():.3f}")

# 9b. Cross-validation with reduced feature sets (10 features each)
print("\n========== CROSS-VALIDATION WITH REDUCED FEATURE SETS (10 FEATURES) ==========")  # Updated comment
datasets = {
    "XGBoost": X_xgb_reduced,
    "Spearman": X_spearman_reduced,
    "Kendall": X_kendall_reduced,
    "MIC": X_mic_reduced
}

for name, dataset in datasets.items():
    scores = cross_val_score(model_cv, dataset, y, cv=cv, scoring='accuracy')
    print(f"{name} reduced set accuracy: {scores.mean():.3f}±{scores.std():.3f}")

# 10. Feature importances
def print_sorted_importance(importance_dict, n=None):
    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    if n:
        sorted_imp = sorted_imp[:n]
    return sorted_imp

# Feature importances from full dataset
print("\n========== FEATURE IMPORTANCES FROM FULL DATASET ==========")
print("\nXGBoost feature importances:")
for feat, imp in print_sorted_importance(xgb_importance, 10):  # Changed from 11 to 10
    print(f"  {feat:20s} {imp:.3f}")

print("\nSpearman feature importances:")
for feat, imp in print_sorted_importance(spearman_importance, 10):  # Changed from 11 to 10
    print(f"  {feat:20s} {imp:.3f}")

print("\nKendall feature importances:")
for feat, imp in print_sorted_importance(kendall_importance, 10):  # Changed from 11 to 10
    print(f"  {feat:20s} {imp:.3f}")

print("\nMIC feature importances:")
for feat, imp in print_sorted_importance(mic_importance, 10):  # Changed from 11 to 10
    print(f"  {feat:20s} {imp:.3f}")

# 11. Show only top 8 feature importances for full and reduced sets
print("\n========== TOP 8 FEATURE IMPORTANCES COMPARISON ==========")

# XGBoost
print("\n--- XGBoost Feature Importance ---")
print("Top 8 Features (Full Set):")
for feat, imp in print_sorted_importance(xgb_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Calculate XGBoost importance on reduced set
_, xgb_reduced_importance = select_top_features(X_xgb_reduced, y, method_name='xgboost')
print("\nTop 8 Features (Reduced Set):")
for feat, imp in print_sorted_importance(xgb_reduced_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Spearman
print("\n--- Spearman Correlation Importance ---")
print("Top 8 Features (Full Set):")
for feat, imp in print_sorted_importance(spearman_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Calculate Spearman importance on reduced set
_, spearman_reduced_importance = select_top_features(X_spearman_reduced, y, method_name='spearman')
print("\nTop 8 Features (Reduced Set):")
for feat, imp in print_sorted_importance(spearman_reduced_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Kendall
print("\n--- Kendall Correlation Importance ---")
print("Top 8 Features (Full Set):")
for feat, imp in print_sorted_importance(kendall_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Calculate Kendall importance on reduced set
_, kendall_reduced_importance = select_top_features(X_kendall_reduced, y, method_name='kendall')
print("\nTop 8 Features (Reduced Set):")
for feat, imp in print_sorted_importance(kendall_reduced_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# MIC
print("\n--- MIC (Maximal Information Coefficient) Importance ---")
print("Top 8 Features (Full Set):")
for feat, imp in print_sorted_importance(mic_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")

# Calculate MIC importance on reduced set
_, mic_reduced_importance = select_top_features(X_mic_reduced, y, method_name='mic')
print("\nTop 8 Features (Reduced Set):")
for feat, imp in print_sorted_importance(mic_reduced_importance, 8):
    print(f"  {feat:20s} {imp:.3f}")
