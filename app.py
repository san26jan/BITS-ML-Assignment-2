import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

with open(ROOT / "feature_names.json", encoding="utf-8") as f:
    FEATURES = json.load(f)

st.set_page_config(page_title="WILP AIML - Classifier Evaluation", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification – Model Evaluation")
st.caption("Reference implementation for BITS WILP M.Tech (AIML) Machine Learning Assignment 2.")
st.info("Upload the held-out test CSV. The app loads pre-trained models and evaluates them; it does not retrain on uploaded data.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
model_name = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))

if uploaded is None:
    st.write("Use the provided `test_data.csv` file to view the evaluation.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
if missing:
    st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

try:
    X_test = data[FEATURES].apply(pd.to_numeric, errors="raise")
    y_test = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("target must contain only 0 (Benign) or 1 (Malignant).")
    model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else None,
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    st.subheader(f"Results: {model_name}")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, "N/A" if value is None else f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        st.dataframe(pd.DataFrame(cm, index=["Actual Benign (0)", "Actual Malignant (1)"],
                                  columns=["Predicted Benign (0)", "Predicted Malignant (1)"]))
    with right:
        st.subheader("Classification Report")
        report = classification_report(y_test, predictions, labels=[0, 1],
                                       target_names=["Benign", "Malignant"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose().round(4))

    with st.expander("Preview uploaded data"):
        st.dataframe(data.head(20))
    output = data.copy()
    output["predicted_target"] = predictions
    output["predicted_label"] = ["Malignant" if p == 1 else "Benign" for p in predictions]
    st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")
except Exception as exc:
    st.error(f"Evaluation failed. Check numeric feature values and target labels. Details: {exc}")
