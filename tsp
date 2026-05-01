# ================================================================
# TITANIC SURVIVAL PREDICTOR
# Author  : Rajan (customise your name here)
# Dataset : Titanic passenger records
# Goal    : Binary classification — Survived (1) / Died (0)
# ================================================================

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# 1. PULL IN THE DATA
# ----------------------------------------------------------------
raw = sns.load_dataset("titanic")
print("Rows & Columns :", raw.shape)
print("\nFirst look at data :")
print(raw.head(3).to_string())

# ----------------------------------------------------------------
# 2. KEEP ONLY USEFUL COLUMNS
# ----------------------------------------------------------------
keep = ["survived", "pclass", "sex", "age",
        "sibsp", "parch", "fare", "embark_town"]
data = raw[keep].copy()

# ----------------------------------------------------------------
# 3. FIX MISSING VALUES
# ----------------------------------------------------------------
data["age"]        = data["age"].fillna(data["age"].median())
data["fare"]       = data["fare"].fillna(data["fare"].median())
data["embark_town"]= data["embark_town"].fillna("Southampton")

print("\nAny nulls left? :", data.isnull().sum().any())

# ----------------------------------------------------------------
# 4. CREATE NEW FEATURES (feature engineering)
# ----------------------------------------------------------------
data["family_total"]  = data["sibsp"] + data["parch"]
data["solo_traveller"]= (data["family_total"] == 0).astype(int)
data["fare_bucket"]   = pd.qcut(data["fare"], q=4, labels=[1, 2, 3, 4])

# ----------------------------------------------------------------
# 5. ENCODE TEXT COLUMNS → NUMBERS
# ----------------------------------------------------------------
data["sex_code"]    = data["sex"].map({"male": 0, "female": 1})
data["port_code"]   = data["embark_town"].map(
                          {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2})

# Drop original text columns
data.drop(columns=["sex", "embark_town"], inplace=True)

# ----------------------------------------------------------------
# 6. QUICK EDA CHARTS
# ----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Titanic — Survival Patterns", fontsize=14, fontweight="bold")

# Chart A — survival by gender
data.groupby("sex_code")["survived"].mean().rename(
    index={0: "Male", 1: "Female"}).plot(
    kind="bar", ax=ax[0], color=["#4c72b0", "#dd8452"], rot=0)
ax[0].set_title("Survival Rate by Gender")
ax[0].set_ylabel("Survival Rate")

# Chart B — survival by ticket class
data.groupby("pclass")["survived"].mean().plot(
    kind="bar", ax=ax[1], color=["#55a868", "#c44e52", "#8172b2"], rot=0)
ax[1].set_title("Survival Rate by Class")

# Chart C — age histogram split by outcome
for outcome, colour, tag in zip([0, 1], ["#c44e52", "#55a868"],
                                 ["Died", "Survived"]):
    data[data["survived"] == outcome]["age"].plot(
        kind="hist", ax=ax[2], alpha=0.55, color=colour,
        label=tag, bins=25)
ax[2].set_title("Age Distribution")
ax[2].legend()

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=150)
plt.show()
print("\nEDA chart saved → titanic_eda.png")

# ----------------------------------------------------------------
# 7. SPLIT INTO TRAINING & TEST SETS
# ----------------------------------------------------------------
feature_cols = ["pclass", "age", "sibsp", "parch", "fare",
                "family_total", "solo_traveller", "fare_bucket",
                "sex_code", "port_code"]

X = data[feature_cols].astype(float)
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y)

print(f"\nTraining samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")

# ----------------------------------------------------------------
# 8. SCALE FEATURES (helps Logistic Regression)
# ----------------------------------------------------------------
scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ----------------------------------------------------------------
# 9. TRAIN THREE CLASSIFIERS
# ----------------------------------------------------------------
classifiers = {
    "Logistic Regression" : LogisticRegression(C=1.0, max_iter=500,
                                               random_state=7),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=6,
                                                   min_samples_leaf=10,
                                                   random_state=7),
    "Random Forest"       : RandomForestClassifier(n_estimators=150,
                                                   max_depth=8,
                                                   random_state=7)
}

scores   = {}
all_preds= {}

for clf_name, clf in classifiers.items():
    # Logistic Regression needs scaled data
    if clf_name == "Logistic Regression":
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
    else:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    scores[clf_name]    = acc
    all_preds[clf_name] = preds

    print(f"\n{'─'*45}")
    print(f"  {clf_name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, preds,
          target_names=["Died", "Survived"]))

# ----------------------------------------------------------------
# 10. PICK WINNER & DRAW CONFUSION MATRIX
# ----------------------------------------------------------------
winner     = max(scores, key=scores.get)
best_preds = all_preds[winner]

print(f"\n🏆 Best model : {winner}  ({scores[winner]*100:.2f}%)")

fig2, ax2 = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"])
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title(f"Confusion Matrix — {winner}", fontweight="bold")
plt.tight_layout()
plt.savefig("titanic_confusion.png", dpi=150)
plt.show()
print("Confusion matrix saved → titanic_confusion.png")

# ----------------------------------------------------------------
# 11. FEATURE IMPORTANCE FROM RANDOM FOREST
# ----------------------------------------------------------------
rf_clf = classifiers["Random Forest"]
imp_df = (pd.DataFrame({
              "Feature"   : feature_cols,
              "Importance": rf_clf.feature_importances_})
            .sort_values("Importance", ascending=True))

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.barh(imp_df["Feature"], imp_df["Importance"],
         color=plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_df))))
ax3.set_title("Feature Importance — Random Forest", fontweight="bold")
ax3.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("titanic_features.png", dpi=150)
plt.show()
print("Feature importance chart saved → titanic_features.png")

# ----------------------------------------------------------------
# 12. MODEL ACCURACY COMPARISON BAR
# ----------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(6, 4))
clr = ["#4c72b0", "#dd8452", "#55a868"]
ax4.bar(scores.keys(), [v*100 for v in scores.values()],
        color=clr, width=0.45)
ax4.set_ylim(70, 100)
ax4.set_ylabel("Accuracy (%)")
ax4.set_title("Model Accuracy Comparison", fontweight="bold")
for i, (k, v) in enumerate(scores.items()):
    ax4.text(i, v*100 + 0.4, f"{v*100:.1f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("titanic_model_compare.png", dpi=150)
plt.show()
print("Model comparison chart saved → titanic_model_compare.png")

print("\n✅ Titanic project complete! All charts saved.")
