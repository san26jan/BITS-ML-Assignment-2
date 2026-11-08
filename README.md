# Breast Cancer Classification – BITS WILP AIML Assignment 2

> **Learning-support reference project.** Run the code yourself, inspect the results, customize the interface/analysis, and update the repository and deployment links before submitting. Do not submit unverified links or screenshots.

## a. Problem statement

Build a binary classification system that predicts whether a breast tumour is benign or malignant from numerical diagnostic measurements. The project compares five classifiers and provides an interactive Streamlit application to evaluate saved models on a labelled test CSV.

## b. Dataset description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569
- **Input features:** 30 numerical features
- **Target:** `target` (0 = Benign, 1 = Malignant)
- **Task:** Binary classification
- **Split:** Stratified 80:20 train/test split, `random_state=42`
- **Positive class for Precision, Recall, F1 and AUC:** Malignant (1)

The dataset is distributed through `sklearn.datasets.load_breast_cancer()` and is based on the UCI dataset. The training script converts the original scikit-learn target encoding so that malignant is the positive class.

## c. GitHub Repository Link

**Replace this placeholder after creating and pushing your own repository:**

`[PASTE YOUR PUBLIC GITHUB REPOSITORY URL HERE]`

The repository should contain `app.py`, `train.py`, `requirements.txt`, this `README.md`, `test_data.csv`, `evaluation_metrics.csv`, `feature_names.json`, and the five saved files under `model/`.

## d. Models used and evaluation

The five required models are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). Standardization is used in the Logistic Regression, KNN, and SVM pipelines. The Decision Tree and Gaussian Naive Bayes models are trained without feature standardization.

### Evaluation metrics

The values below are produced by the included `train.py` using the fixed stratified split. Re-run the script in your own environment and use your own run's output if it differs.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on model performance

| Model | Observation |
|---|---|
| Logistic Regression | Strong overall performance and the highest AUC (0.9960), indicating excellent ranking/discrimination. It also has high precision, recall and MCC. |
| Decision Tree | Easiest to interpret, but it has the lowest accuracy and AUC in this experiment. A single tree can be sensitive to the particular split. |
| KNN | Performs well after feature standardization. Its accuracy is strong, although recall is lower than Logistic Regression and SVM on this test split. |
| Gaussian Naive Bayes | Achieves perfect precision on this test split but lower recall (0.8333), meaning some malignant cases are missed. Its conditional-independence assumption may limit performance. |
| Support Vector Machine | Best accuracy (0.9737), F1 (0.9630) and MCC (0.9442) in this run. It is the overall winner on the selected split, while Logistic Regression has the highest AUC. |

**Overall winner:** Support Vector Machine, based on the highest accuracy, F1 and MCC in this test run. Logistic Regression is a close competitor and has the highest AUC. The winner is specific to this dataset split and should not be treated as a universal conclusion.

## Streamlit application

The application provides:
1. CSV test-data upload.
2. A model-selection dropdown for all five saved classifiers.
3. Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix and classification report.
5. A downloadable CSV containing predictions.

The app requires the uploaded CSV to contain all 30 feature columns and a `target` column. It loads saved models from `model/` and **does not retrain** on uploaded test data.

## Run locally

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal and upload `test_data.csv`.

## Deploy on Streamlit Community Cloud

1. Push the complete project folder to a public GitHub repository.
2. Visit https://streamlit.io/cloud and sign in with GitHub.
3. Choose **New app**, select your repository and branch, and set the main file to `app.py`.
4. Deploy and wait for the app to finish building.
5. Open the public URL in a private/incognito browser window and test the CSV upload and each model.

**Live Streamlit App Link:** `[PASTE YOUR DEPLOYED STREAMLIT URL HERE]`

## Reproducibility and files

- `train.py`: loads the dataset, splits it, trains all five models, evaluates them and saves artifacts.
- `model/`: five fitted model pipelines/classifiers.
- `test_data.csv`: held-out test set with target labels; upload this in the app.
- `evaluation_metrics.csv`: metrics from the included fixed split.
- `app.py`: Streamlit evaluation interface.

## Academic integrity and final checks

This is a reference implementation to support learning. Review the code, understand the metric calculations, customize the app and write observations in your own words. Keep your own Git commit history. Perform the execution on BITS Virtual Lab and capture your actual screenshot; do not use a simulated screenshot. Replace both link placeholders with working URLs before PDF submission.
