"""Train the five required classifiers and export held-out test data + metrics.

Dataset: Breast Cancer Wisconsin (Diagnostic), distributed with scikit-learn
and originally available from the UCI Machine Learning Repository.
Target convention in this project: 0 = Benign, 1 = Malignant.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# scikit-learn encodes malignant as 0 and benign as 1; invert to make
# malignant the positive class for this assignment's metric reporting.
y = (dataset.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
    ]),
}
filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

metric_rows, confusion_matrices = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predicted),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predicted, zero_division=0),
        "Recall": recall_score(y_test, predicted, zero_division=0),
        "F1": f1_score(y_test, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted),
    })
    confusion_matrices[name] = confusion_matrix(y_test, predicted).tolist()
    joblib.dump(model, MODEL_DIR / filenames[name])

test_data = X_test.copy()
test_data["target"] = np.asarray(y_test)
test_data.to_csv(ROOT / "test_data.csv", index=False)
pd.DataFrame(metric_rows).to_csv(ROOT / "evaluation_metrics.csv", index=False)
(ROOT / "feature_names.json").write_text(json.dumps(list(X.columns), indent=2))
(ROOT / "confusion_matrices.json").write_text(json.dumps(confusion_matrices, indent=2))
(ROOT / "dataset_info.json").write_text(json.dumps({
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "repository": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    "instances": int(len(X)),
    "input_features": int(X.shape[1]),
    "target": {"0": "Benign", "1": "Malignant"},
    "positive_class_for_metrics": "Malignant (1)",
    "split": "80% train / 20% test, stratified, random_state=42",
}, indent=2))

print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
print("\\nSaved models, test_data.csv and evaluation_metrics.csv.")
