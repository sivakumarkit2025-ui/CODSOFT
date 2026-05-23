# ============================================================
#   Iris Flower Classification Using Machine Learning
#   Author  : [Your Name]
#   Purpose : Classify iris flowers into species based on
#             sepal and petal measurements
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets            import load_iris
from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.preprocessing       import StandardScaler
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.svm                 import SVC
from sklearn.metrics             import (classification_report,
                                          confusion_matrix,
                                          accuracy_score)


# ─────────────────────────────────────────
#  STEP 1 ── Load the Iris Dataset
# ─────────────────────────────────────────
# The Iris dataset is built into scikit-learn.
# No download needed — loads instantly.

iris      = load_iris()
df        = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map({
    0: "Setosa", 1: "Versicolor", 2: "Virginica"
})

print("=" * 58)
print("  IRIS DATASET OVERVIEW")
print("=" * 58)
print(df.head(10).to_string(index=False))
print(f"\n  Total Samples  : {len(df)}")
print(f"  Features       : {list(iris.feature_names)}")
print(f"  Species        : Setosa, Versicolor, Virginica")
print(f"  Null Values    : {df.isnull().sum().sum()}")
print("\n  Samples per Species :")
print(df["species_name"].value_counts().to_string())
print("\n  Basic Statistics :")
print(df.describe().round(2))


# ─────────────────────────────────────────
#  STEP 2 ── Exploratory Data Analysis
# ─────────────────────────────────────────

species_colors = {"Setosa": "steelblue",
                  "Versicolor": "tomato",
                  "Virginica": "seagreen"}

# Pairplot — see how species separate across all features
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Feature Distribution by Species", fontsize=14, fontweight="bold")

feature_pairs = [
    ("sepal length (cm)", "sepal width (cm)"),
    ("sepal length (cm)", "petal length (cm)"),
    ("petal length (cm)", "petal width (cm)"),
    ("sepal width (cm)",  "petal width (cm)"),
]

for ax, (fx, fy) in zip(axes.flatten(), feature_pairs):
    for species, color in species_colors.items():
        subset = df[df["species_name"] == species]
        ax.scatter(subset[fx], subset[fy],
                   label=species, color=color,
                   alpha=0.7, edgecolors="white", s=55)
    ax.set_xlabel(fx)
    ax.set_ylabel(fy)
    ax.set_title(f"{fx.split()[0].title()} vs {fy.split()[0].title()}")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("iris_eda_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  ✔  EDA scatter plots saved → iris_eda_scatter.png")

# Box plots — feature spread per species
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Feature Box Plots by Species", fontsize=13, fontweight="bold")

palette = ["steelblue", "tomato", "seagreen"]
for ax, feature in zip(axes, iris.feature_names):
    sns.boxplot(data=df, x="species_name", y=feature,
                palette=palette, ax=ax)
    ax.set_title(feature.replace(" (cm)", "").title())
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Box plots saved → iris_boxplots.png")

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop(columns=["species", "species_name"]).corr(numeric_only=True),
            annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("iris_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Correlation heatmap saved → iris_correlation.png")


# ─────────────────────────────────────────
#  STEP 3 ── Feature & Label Split
# ─────────────────────────────────────────

X = df[list(iris.feature_names)]   # sepal/petal measurements
y = df["species"]                   # 0=Setosa, 1=Versicolor, 2=Virginica

print("\n  ✔  Features : sepal length, sepal width, petal length, petal width")
print("  ✔  Target   : species (0=Setosa, 1=Versicolor, 2=Virginica)")


# ─────────────────────────────────────────
#  STEP 4 ── Train / Test Split  (80 / 20)
# ─────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y        # equal species ratio in both splits
)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")


# ─────────────────────────────────────────
#  STEP 5 ── Feature Scaling
# ─────────────────────────────────────────

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("  ✔  Features scaled with StandardScaler")


# ─────────────────────────────────────────
#  STEP 6 ── Train Multiple Models
# ─────────────────────────────────────────

models = {
    "Logistic Regression" : LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine" : SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

print("\n" + "=" * 58)
print("  MODEL TRAINING & EVALUATION")
print("=" * 58)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred   = model.predict(X_test_sc)
    acc      = accuracy_score(y_test, y_pred)
    cv_score = cross_val_score(model, X_train_sc, y_train,
                                cv=5, scoring="accuracy").mean()

    results[name] = {"accuracy": acc, "cv": cv_score, "pred": y_pred}

    print(f"\n  ── {name} ──")
    print(f"     Test Accuracy     : {acc*100:.2f}%")
    print(f"     Cross-Val (5-fold): {cv_score*100:.2f}%")
    print(classification_report(y_test, y_pred,
                                 target_names=iris.target_names,
                                 zero_division=0))


# ─────────────────────────────────────────
#  STEP 7 ── Confusion Matrices (All Models)
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.1f}%")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Confusion matrices saved → iris_confusion_matrices.png")


# ─────────────────────────────────────────
#  STEP 8 ── Model Accuracy Comparison Bar Chart
# ─────────────────────────────────────────

model_names = list(results.keys())
accuracies  = [v["accuracy"] * 100 for v in results.values()]
cv_scores   = [v["cv"] * 100 for v in results.values()]

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, accuracies, width,
               label="Test Accuracy", color="steelblue", edgecolor="black")
bars2 = ax.bar(x + width/2, cv_scores,  width,
               label="CV Accuracy",   color="tomato",    edgecolor="black")

ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy Comparison")
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10)
ax.set_ylim([80, 105])
ax.legend()
ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("iris_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  Model comparison chart saved → iris_model_comparison.png")


# ─────────────────────────────────────────
#  STEP 9 ── Predict a New Iris Flower
# ─────────────────────────────────────────

print("\n" + "=" * 58)
print("  PREDICT A NEW IRIS FLOWER SAMPLE")
print("=" * 58)

new_flower = pd.DataFrame({
    "sepal length (cm)" : [5.1],
    "sepal width (cm)"  : [3.5],
    "petal length (cm)" : [1.4],
    "petal width (cm)"  : [0.2],
})

new_scaled  = scaler.transform(new_flower)
best_model  = models["Random Forest"]
prediction  = best_model.predict(new_scaled)[0]
species_out = iris.target_names[prediction]

print(f"\n  Input  → Sepal: 5.1cm x 3.5cm | Petal: 1.4cm x 0.2cm")
print(f"  Result → Predicted Species : {species_out.upper()}")
print("\n  ✔  All done! Push results to GitHub.")