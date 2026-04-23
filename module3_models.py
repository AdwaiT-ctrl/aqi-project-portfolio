"""Module 3 model experiments for the AQI portfolio.

The script uses the attached AQI dataset when it is present in the workspace.
It keeps a fallback dataset so the file can still run if the CSV is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


RANDOM_STATE = 42
DATA_PATH = Path("city_day.csv")
AQI_THRESHOLD = 100
FEATURE_COLUMNS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]


def load_project_frame() -> tuple[pd.DataFrame, str, list[str]]:
    """Load AQI data if it exists; otherwise fall back to a clean binary dataset."""
    if DATA_PATH.exists():
        frame = pd.read_csv(DATA_PATH)
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        numeric_columns = [column for column in FEATURE_COLUMNS + ["AQI"] if column in frame.columns]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=numeric_columns).copy()
        frame["label"] = (frame["AQI"] >= AQI_THRESHOLD).astype(int)
        features = [column for column in FEATURE_COLUMNS if column in frame.columns]
        return frame, "label", features

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()
    frame["label"] = dataset.target
    features = list(dataset.feature_names)
    return frame, "label", features


def split_features_labels(frame: pd.DataFrame, label_column: str, features: list[str]):
    X = frame[features].copy()
    y = frame[label_column].astype(int).copy()
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "name": name,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, zero_division=0),
    }


def run_all_models():
    frame, label_column, features = load_project_frame()
    X_train, X_test, y_train, y_test = split_features_labels(frame, label_column, features)
    median_threshold = np.nanmedian(X_train.values, axis=0)

    models = {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        ),
        "Gaussian NB": make_pipeline(
            SimpleImputer(strategy="median"),
            GaussianNB(),
        ),
        "Multinomial NB": make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(lambda data: np.clip(data, a_min=0, a_max=None), validate=False),
            MultinomialNB(),
        ),
        "Bernoulli NB": make_pipeline(
            SimpleImputer(strategy="median"),
            FunctionTransformer(lambda data: (data > median_threshold).astype(int), validate=False),
            BernoulliNB(),
        ),
        "Categorical NB": make_pipeline(
            SimpleImputer(strategy="median"),
            KBinsDiscretizer(
                n_bins=4,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
            ),
            FunctionTransformer(lambda data: data.astype(int), validate=False),
            CategoricalNB(),
        ),
        "Support Vector Machine": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
        ),
    }

    results = [
        evaluate_model(name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    ]

    return frame, X_train, X_test, y_train, y_test, results


if __name__ == "__main__":
    frame, X_train, X_test, y_train, y_test, results = run_all_models()
    print(f"Rows: {len(frame)}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    for result in results:
        print("\n" + result["name"])
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(result["confusion_matrix"])
        print(result["report"])
