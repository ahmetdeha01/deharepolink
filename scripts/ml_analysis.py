"""
DSA 210 - ML Analysis
Predicting 3D Print Quality: roughness, tensile_strength, elongation
Models: Linear Regression, Random Forest, XGBoost, SVR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

#  1. Load Data 
df = pd.read_csv('enriched_data.csv')

# Drop raw categorical columns (already encoded)
df = df.drop(columns=['infill_pattern', 'material'])

FEATURES = [c for c in df.columns if c not in ['roughness', 'tensile_strength', 'elongation']]
TARGETS   = ['roughness', 'tensile_strength', 'elongation']

X = df[FEATURES]
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Targets:  {TARGETS}")
print(f"Dataset:  {X.shape[0]} rows\n")

# ── 2. Models 
models = {
    'Linear Regression' : LinearRegression(),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost'           : XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'SVR'               : SVR(kernel='rbf', C=10, epsilon=0.1),
}

# ── 3. Train / Evaluate 
results = {}   # results[target][model_name] = {rmse, mae, r2, cv_r2}

for target in TARGETS:
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale for SVR and Linear Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results[target] = {}

    for name, model in models.items():
        # SVR and Linear Regression use scaled data
        if name in ['SVR', 'Linear Regression']:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        results[target][name] = {
            'RMSE'   : round(rmse, 3),
            'MAE'    : round(mae, 3),
            'R2'     : round(r2, 3),
            'CV_R2'  : round(cv_scores.mean(), 3),
        }

#  4. Print Results Table 
for target in TARGETS:
    print(f"\n{'='*60}")
    print(f"  TARGET: {target.upper()}")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV R²':>8}")
    print(f"  {'-'*54}")
    for name, m in results[target].items():
        print(f"  {name:<22} {m['RMSE']:>8} {m['MAE']:>8} {m['R2']:>8} {m['CV_R2']:>8}")

#  5. Feature Importance (Random Forest) 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Importance — Random Forest', fontsize=15, fontweight='bold')

for ax, target in zip(axes, TARGETS):
    y = df[target]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    colors = ['#2ecc71' if v >= imp.quantile(0.66) else
              '#f39c12' if v >= imp.quantile(0.33) else '#e74c3c' for v in imp]

    imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(target, fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.axvline(imp.mean(), color='black', linestyle='--', linewidth=0.8, label='mean')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: feature_importance.png")

#  6. Model Comparison Plot 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison — R² Score (Test Set)', fontsize=15, fontweight='bold')

model_names = list(models.keys())
colors_bar  = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

for ax, target in zip(axes, TARGETS):
    r2_scores = [results[target][m]['R2'] for m in model_names]
    bars = ax.bar(model_names, r2_scores, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_title(target, fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_ylim(min(0, min(r2_scores)) - 0.05, 1.05)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: model_comparison.png")

#  7. Actual vs Predicted (Best Model per Target) 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Actual vs Predicted — Best Model per Target', fontsize=14, fontweight='bold')

for ax, target in zip(axes, TARGETS):
    # Pick best model by R2
    best_name = max(results[target], key=lambda m: results[target][m]['R2'])
    best_r2   = results[target][best_name]['R2']

    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models[best_name]
    if best_name in ['SVR', 'Linear Regression']:
        scaler = StandardScaler()
        model.fit(scaler.fit_transform(X_train), y_train)
        y_pred = model.predict(scaler.transform(X_test))
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    ax.scatter(y_test, y_pred, alpha=0.7, color='#3498db', edgecolors='white', linewidth=0.3)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{target}\n{best_name} (R²={best_r2})', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: actual_vs_predicted.png")

print("\n🎉 ML analysis complete!")
