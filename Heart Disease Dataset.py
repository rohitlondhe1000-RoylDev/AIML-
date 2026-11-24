import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns, na_values="?")

df["ca"] = df["ca"].fillna(df["ca"].median())
df["thal"] = df["thal"].fillna(df["thal"].mode()[0])

df["target"] = (df["target"] > 0).astype(int)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))
dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])

rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_model.predict(X_test))
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print("--- Decision Tree Performance ---")
print(f"Training Accuracy : {dt_train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {dt_test_acc * 100:.2f}%")
print(f"ROC-AUC Score     : {dt_auc:.4f}")

print("\n--- Random Forest Performance ---")
print(f"Training Accuracy : {rf_train_acc * 100:.2f}%")
print(f"Testing Accuracy  : {rf_test_acc * 100:.2f}%")
print(f"ROC-AUC Score     : {rf_auc:.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test), target_names=["No Disease", "Heart Disease"]))

depths = range(1, 15)
dt_train_scores, dt_test_scores = [], []
rf_train_scores, rf_test_scores = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    dt_train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    dt_test_scores.append(accuracy_score(y_test, dt.predict(X_test)))

    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42).fit(X_train, y_train)
    rf_train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    rf_test_scores.append(accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(depths, dt_train_scores, label="DT Train Accuracy", marker="o", linestyle="--")
plt.plot(depths, dt_test_scores, label="DT Test Accuracy", marker="o")
plt.plot(depths, rf_train_scores, label="RF Train Accuracy", marker="s", linestyle="--")
plt.plot(depths, rf_test_scores, label="RF Test Accuracy", marker="s")
plt.xlabel("Tree Max Depth (Model Complexity)")
plt.ylabel("Accuracy")
plt.title("Bias-Variance Tradeoff Across Tree Depth")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.subplot(1, 2, 2)
importances.plot(kind="barh", color="teal")
plt.xlabel("Mean Decrease in Impurity (Gini Importance)")
plt.title("Random Forest Feature Importance")
plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()



#--- Decision Tree Performance ---
#Training Accuracy : 100.00%
#Testing Accuracy  : 73.77%
#ROC-AUC Score     : 0.7440

#--- Random Forest Performance ---
#Training Accuracy : 100.00%
#Testing Accuracy  : 88.52%
#ROC-AUC Score     : 0.9518

#Random Forest Classification Report:
#               precision    recall  f1-score   support
#
#   No Disease       0.93      0.85      0.89        33
#Heart Disease       0.84      0.93      0.88        28

#     accuracy                           0.89        61
#    macro avg       0.89      0.89      0.89        61
# weighted avg       0.89      0.89      0.89        61