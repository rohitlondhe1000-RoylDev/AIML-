import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

print("Gaussian Naive Bayes Evaluation Metrics:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.title("Confusion Matrix - Gaussian Naive Bayes (Iris)")
plt.tight_layout()
plt.show()




#Gaussian Naive Bayes Evaluation Metrics:
#Accuracy  : 96.67%
#Precision : 96.97%
#Recall    : 96.67%

#Detailed Classification Report:
#              precision    recall  f1-score   support
#
#      setosa       1.00      1.00      1.00        10
#  versicolor       1.00      0.90      0.95        10
#   virginica       0.91      1.00      0.95        10
#
#    accuracy                           0.97        30
#   macro avg       0.97      0.97      0.97        30
#weighted avg       0.97      0.97      0.97        30