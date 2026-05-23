# ============================================================
#   Credit Card Fraud Detection — ML Pipeline
#   Author  : [Your Name]
#   Purpose : Classify transactions as fraudulent or genuine
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection     import train_test_split
from sklearn.preprocessing       import StandardScaler
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.metrics             import (classification_report,
                                          confusion_matrix,
                                          roc_auc_score,
                                          roc_curve)
from imblearn.over_sampling      import SMOTE


# ─────────────────────────────────────────
#  STEP 1 ── Load the dataset
# ─────────────────────────────────────────
# Download creditcard.csv from:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Then place it in the same folder as this script.

df = pd.read_csv("C:\\Users\\ADMIN\\Downloads\\creditcard.csv")
print("=" * 55)
print("  DATASET OVERVIEW")
print("=" * 55)
print(f"  Total transactions : {len(df):,}")
print(f"  Fraudulent         : {df['Class'].sum():,}")
print(f"  Genuine            : {(df['Class'] == 0).sum():,}")
print(f"  Fraud rate         : {df['Class'].mean()*100:.4f}%")
print("=" * 55)


# ─────────────────────────────────────────
#  STEP 2 ── Preprocessing & Normalisation
# ─────────────────────────────────────────
# 'Amount' and 'Time' are the only raw (non-PCA) columns.
# We scale them to the same range as the V1-V28 features.

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])

# Drop the originals — keep only the processed versions
df.drop(columns=["Amount", "Time"], inplace=True)

print("\n  ✔  Amount and Time columns normalised")


# ─────────────────────────────────────────
#  STEP 3 ── Feature / Label split
# ─────────────────────────────────────────

X = df.drop(columns=["Class"])   # all feature columns
y = df["Class"]                   # 0 = genuine, 1 = fraud


# ─────────────────────────────────────────
#  STEP 4 ── Train / Test split  (80 / 20)
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y          # keeps fraud ratio equal in both splits
)

print(f"\n  Training samples : {len(X_train):,}")
print(f"  Testing  samples : {len(X_test):,}")


# ─────────────────────────────────────────
#  STEP 5 ── Handle Class Imbalance (SMOTE)
# ─────────────────────────────────────────
# SMOTE creates synthetic minority-class (fraud) samples
# so the model learns fraud patterns properly.

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print(f"\n  After SMOTE — fraud samples  : {(y_train_bal == 1).sum():,}")
print(f"  After SMOTE — genuine samples: {(y_train_bal == 0).sum():,}")
print("  ✔  Class imbalance handled\n")


# ─────────────────────────────────────────
#  STEP 6 ── Train Models
# ─────────────────────────────────────────

# --- Model A : Logistic Regression ---
print("  Training Logistic Regression …")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_bal, y_train_bal)

# --- Model B : Random Forest ---
print("  Training Random Forest …\n")
rf_model = RandomForestClassifier(
    n_estimators = 100,
    random_state = 42,
    n_jobs       = -1      # use all CPU cores
)
rf_model.fit(X_train_bal, y_train_bal)


# ─────────────────────────────────────────
#  STEP 7 ── Evaluate Both Models
# ─────────────────────────────────────────

def evaluate(model, name, X_test, y_test):
    """Print classification metrics and return predictions."""
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]
    roc_score  = roc_auc_score(y_test, y_prob)

    print("=" * 55)
    print(f"  MODEL : {name}")
    print("=" * 55)
    print(classification_report(y_test, y_pred,
                                 target_names=["Genuine", "Fraud"]))
    print(f"  ROC-AUC Score : {roc_score:.4f}")
    print()
    return y_pred, y_prob


lr_pred, lr_prob = evaluate(lr_model, "Logistic Regression", X_test, y_test)
rf_pred, rf_prob = evaluate(rf_model, "Random Forest",       X_test, y_test)


# ─────────────────────────────────────────
#  STEP 8 ── Visualisations
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Credit Card Fraud Detection — Results", fontsize=14, fontweight="bold")

# ── Plot 1 : Class distribution (before SMOTE) ──
counts = y.value_counts()
axes[0].bar(["Genuine", "Fraud"], counts.values,
            color=["steelblue", "tomato"], edgecolor="black")
axes[0].set_title("Transaction Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

# ── Plot 2 : Confusion Matrix (Random Forest) ──
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Genuine", "Fraud"],
            yticklabels=["Genuine", "Fraud"],
            ax=axes[1])
axes[1].set_title("Confusion Matrix — Random Forest")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# ── Plot 3 : ROC Curves ──
for prob, name, color in [
    (lr_prob, "Logistic Regression", "royalblue"),
    (rf_prob, "Random Forest",       "tomato"),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val      = roc_auc_score(y_test, prob)
    axes[2].plot(fpr, tpr, label=f"{name}  (AUC={auc_val:.3f})", color=color)

axes[2].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[2].set_title("ROC Curve Comparison")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.savefig("fraud_detection_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Chart saved → fraud_detection_results.png")


# ─────────────────────────────────────────
#  STEP 9 ── Feature Importance (Random Forest)
# ─────────────────────────────────────────

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10       = importances.nlargest(10).sort_values()

plt.figure(figsize=(8, 5))
top10.plot(kind="barh", color="steelblue", edgecolor="black")
plt.title("Top 10 Important Features — Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Feature importance chart saved → feature_importance.png")

print("\n  ✔  All done! Review the metrics and charts above.")
