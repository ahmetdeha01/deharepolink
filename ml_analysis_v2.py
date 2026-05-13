"""
DSA 210 - ML Analysis with Hyperparameter Tuning
GridSearchCV + Training/Validation/Test visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv('enriched_data.csv')
df = df.drop(columns=['infill_pattern', 'material'])

FEATURES = [c for c in df.columns if c not in ['roughness', 'tensile_strength', 'elongation']]
TARGETS   = ['roughness', 'tensile_strength', 'elongation']

X = df[FEATURES]
print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")

# ── 2. Hyperparameter Grids ───────────────────────────────────────────────
param_grids = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {'fit_intercept': [True, False]},
        'scale': True
    },
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        },
        'scale': False
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, verbosity=0),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2]
        },
        'scale': False
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear']
        },
        'scale': True
    }
}

# ── 3. Train with GridSearchCV ────────────────────────────────────────────
results = {}   # results[target][model] = {best_params, train_r2, val_r2, test_r2, ...}

for target in TARGETS:
    print(f"\n{'='*60}")
    print(f"  TARGET: {target.upper()}")
    print(f"{'='*60}")

    y = df[target]

    # 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    results[target] = {}

    for name, config in param_grids.items():
        scaler = StandardScaler()

        if config['scale']:
            X_tr = scaler.fit_transform(X_train)
            X_v  = scaler.transform(X_val)
            X_te = scaler.transform(X_test)
        else:
            X_tr, X_v, X_te = X_train.values, X_val.values, X_test.values

        # GridSearchCV (cv=5 on training set)
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
        grid.fit(X_tr, y_train)
        best_model = grid.best_estimator_

        # Scores
        train_r2 = r2_score(y_train, best_model.predict(X_tr))
        val_r2   = r2_score(y_val,   best_model.predict(X_v))
        test_r2  = r2_score(y_test,  best_model.predict(X_te))
        test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_te)))
        test_mae  = mean_absolute_error(y_test, best_model.predict(X_te))

        results[target][name] = {
            'best_params': grid.best_params_,
            'train_r2'   : round(train_r2, 3),
            'val_r2'     : round(val_r2, 3),
            'test_r2'    : round(test_r2, 3),
            'test_rmse'  : round(test_rmse, 3),
            'test_mae'   : round(test_mae, 3),
            'best_model' : best_model,
            'scaler'     : scaler if config['scale'] else None,
            'scale'      : config['scale'],
            'X_train': X_tr, 'y_train': y_train,
            'X_val': X_v,   'y_val': y_val,
            'X_test': X_te, 'y_test': y_test
        }

        print(f"\n  {name}")
        print(f"    Best params : {grid.best_params_}")
        print(f"    Train R²    : {train_r2:.3f}")
        print(f"    Val   R²    : {val_r2:.3f}")
        print(f"    Test  R²    : {test_r2:.3f}  |  RMSE: {test_rmse:.3f}  |  MAE: {test_mae:.3f}")

# ── 4. Plot: Train / Val / Test R² Comparison ─────────────────────────────
model_names = list(param_grids.keys())
colors = {'train_r2': '#3498db', 'val_r2': '#2ecc71', 'test_r2': '#e74c3c'}
x = np.arange(len(model_names))
width = 0.25

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Train / Validation / Test R² — After Hyperparameter Tuning', fontsize=14, fontweight='bold')

for ax, target in zip(axes, TARGETS):
    train_scores = [results[target][m]['train_r2'] for m in model_names]
    val_scores   = [results[target][m]['val_r2']   for m in model_names]
    test_scores  = [results[target][m]['test_r2']  for m in model_names]

    b1 = ax.bar(x - width, train_scores, width, label='Train',      color=colors['train_r2'], alpha=0.85)
    b2 = ax.bar(x,         val_scores,   width, label='Validation', color=colors['val_r2'],   alpha=0.85)
    b3 = ax.bar(x + width, test_scores,  width, label='Test',       color=colors['test_r2'],  alpha=0.85)

    ax.set_title(target, fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score')
    ax.set_xticks(x)
    ax.set_xticklabels(['LR', 'RF', 'XGB', 'SVR'], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('train_val_test_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: train_val_test_comparison.png")

# ── 5. Plot: Learning Curves ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Learning Curves — Best Models After Tuning', fontsize=14, fontweight='bold')

for row, target in enumerate(TARGETS):
    y = df[target]
    for col, name in enumerate(model_names):
        ax = axes[row][col]
        config = param_grids[name]
        best_model = results[target][name]['best_model']

        if config['scale']:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_scaled, y, cv=5, scoring='r2',
            train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        ax.plot(train_sizes, train_mean, 'o-', color='#3498db', label='Train')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#3498db')
        ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Validation')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#e74c3c')

        ax.set_title(f'{name}\n({target})', fontsize=9, fontweight='bold')
        ax.set_xlabel('Training samples', fontsize=8)
        ax.set_ylabel('R²', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: learning_curves.png")

# ── 6. Plot: Best Params Summary ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle('Best Hyperparameters — GridSearchCV Results', fontsize=13, fontweight='bold')

for ax, target in zip(axes, TARGETS):
    rows = []
    for name in model_names:
        params = results[target][name]['best_params']
        param_str = '\n'.join([f"{k}: {v}" for k, v in params.items()])
        test_r2 = results[target][name]['test_r2']
        rows.append([name, param_str, test_r2])

    ax.axis('off')
    table = ax.table(
        cellText=[[r[0], r[1], r[2]] for r in rows],
        colLabels=['Model', 'Best Params', 'Test R²'],
        cellLoc='center', loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 3)
    ax.set_title(target, fontsize=11, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('best_params_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: best_params_table.png")

# ── 7. Plot: Actual vs Predicted (all models, best target) ───────────────
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('Actual vs Predicted — All Models, All Targets', fontsize=14, fontweight='bold')

for row, target in enumerate(TARGETS):
    for col, name in enumerate(model_names):
        ax = axes[row][col]
        r = results[target][name]
        y_pred = r['best_model'].predict(r['X_test'])
        y_true = r['y_test']

        ax.scatter(y_true, y_pred, alpha=0.7, color='#3498db', edgecolors='white', linewidth=0.3, s=40)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1.5)
        ax.set_title(f'{name} | {target}\nR²={r["test_r2"]}', fontsize=8, fontweight='bold')
        ax.set_xlabel('Actual', fontsize=7)
        ax.set_ylabel('Predicted', fontsize=7)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: actual_vs_predicted_all.png")

print("\n🎉 All done! Files saved:")
print("  - train_val_test_comparison.png")
print("  - learning_curves.png")
print("  - best_params_table.png")
print("  - actual_vs_predicted_all.png")
