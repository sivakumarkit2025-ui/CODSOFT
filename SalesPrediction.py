# ============================================================
#   Sales Prediction Using Machine Learning
#   Author  : [Your Name]
#   Purpose : Forecast product sales based on advertising spend
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.preprocessing       import StandardScaler
from sklearn.linear_model        import LinearRegression
from sklearn.ensemble            import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics             import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────
#  STEP 1 ── Load the Dataset
# ─────────────────────────────────────────
# Dataset : Advertising.csv
# Download from :
# https://www.kaggle.com/datasets/bumba5341/advertisingcsv
# Columns : TV, Radio, Newspaper (ad spend) → Sales
#
# OR the code below auto-generates sample data if file not found.

try:
    df = pd.read_csv("Advertising.csv")
    # drop unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    print("  ✔  Loaded Advertising.csv")
except FileNotFoundError:
    print("  ⚠  Advertising.csv not found — generating sample data …")
    np.random.seed(42)
    n = 200
    TV        = np.random.uniform(0.7, 296, n)
    Radio     = np.random.uniform(0.0, 49,  n)
    Newspaper = np.random.uniform(0.3, 114, n)
    Sales     = (0.055 * TV + 0.11 * Radio + 0.01 * Newspaper
                 + np.random.normal(0, 1.2, n) + 4.5)
    df = pd.DataFrame({"TV": TV, "Radio": Radio,
                        "Newspaper": Newspaper, "Sales": Sales})
    print("  ✔  Sample dataset created (200 rows)")

print("\n" + "=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(df.head())
print(f"\n  Shape  : {df.shape}")
print(f"  Nulls  : {df.isnull().sum().sum()}")
print("\n  Basic Statistics :")
print(df.describe().round(2))


# ─────────────────────────────────────────
#  STEP 2 ── Exploratory Data Analysis
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Ad Spend vs Sales — Scatter Plots", fontsize=13, fontweight="bold")

colors = ["steelblue", "tomato", "seagreen"]
for ax, col, color in zip(axes, ["TV", "Radio", "Newspaper"], colors):
    ax.scatter(df[col], df["Sales"], alpha=0.6, color=color, edgecolors="white", s=50)
    ax.set_xlabel(f"{col} Ad Spend ($)")
    ax.set_ylabel("Sales (units)")
    ax.set_title(f"{col} vs Sales")

plt.tight_layout()
plt.savefig("eda_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✔  EDA scatter plots saved → eda_scatter.png")

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Correlation heatmap saved → correlation_heatmap.png")


# ─────────────────────────────────────────
#  STEP 3 ── Feature & Target Split
# ─────────────────────────────────────────

X = df[["TV", "Radio", "Newspaper"]]   # advertising spend features
y = df["Sales"]                          # target : sales volume

print("\n  ✔  Features : TV, Radio, Newspaper")
print("  ✔  Target   : Sales")


# ─────────────────────────────────────────
#  STEP 4 ── Train / Test Split  (80 / 20)
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────
#  STEP 5 ── Feature Scaling
# ─────────────────────────────────────────

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("  ✔  Features scaled with StandardScaler")


# ─────────────────────────────────────────
#  STEP 6 ── Train Multiple Models
# ─────────────────────────────────────────

models = {
    "Linear Regression"        : LinearRegression(),
    "Random Forest Regressor"  : RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting"        : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}

print("\n" + "=" * 55)
print("  MODEL TRAINING & EVALUATION")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="r2").mean()

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv, "pred": y_pred}

    print(f"\n  ── {name} ──")
    print(f"     MAE   (Mean Abs Error)    : {mae:.4f}")
    print(f"     RMSE  (Root Mean Sq Err)  : {rmse:.4f}")
    print(f"     R²    (Accuracy proxy)    : {r2:.4f}")
    print(f"     Cross-Val R²  (5-fold)    : {cv:.4f}")


# ─────────────────────────────────────────
#  STEP 7 ── Visualise Predictions
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Actual vs Predicted Sales — All Models", fontsize=13, fontweight="bold")

palette = ["steelblue", "tomato", "seagreen"]
for ax, (name, res), color in zip(axes, results.items(), palette):
    ax.scatter(y_test, res["pred"], alpha=0.7, color=color, edgecolors="white", s=50)
    lims = [min(y_test.min(), res["pred"].min()),
            max(y_test.max(), res["pred"].max())]
    ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect fit")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"{name}\nR² = {res['R2']:.3f}")
    ax.legend()

plt.tight_layout()
plt.savefig("predictions_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✔  Prediction charts saved → predictions_comparison.png")


# ─────────────────────────────────────────
#  STEP 8 ── Feature Importance (Random Forest)
# ─────────────────────────────────────────

rf_model     = models["Random Forest Regressor"]
importances  = pd.Series(rf_model.feature_importances_, index=X.columns)

plt.figure(figsize=(6, 4))
importances.sort_values().plot(kind="barh", color=["seagreen","tomato","steelblue"],
                                edgecolor="black")
plt.title("Feature Importance — Which Ad Platform Drives Sales Most?")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_sales.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Feature importance chart saved → feature_importance_sales.png")


# ─────────────────────────────────────────
#  STEP 9 ── Predict New Sales (Business Use)
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  SAMPLE BUSINESS PREDICTION")
print("=" * 55)

new_campaign = pd.DataFrame({
    "TV"        : [150.0],   # $150 TV ad spend
    "Radio"     : [25.0],    # $25  Radio ad spend
    "Newspaper" : [10.0]     # $10  Newspaper ad spend
})

new_scaled   = scaler.transform(new_campaign)
best_model   = models["Random Forest Regressor"]
prediction   = best_model.predict(new_scaled)[0]

print(f"\n  Ad Budget  → TV: $150 | Radio: $25 | Newspaper: $10")
print(f"  Predicted Sales → {prediction:.2f} units")
print("\n  ✔  All done! Review metrics and charts above.")