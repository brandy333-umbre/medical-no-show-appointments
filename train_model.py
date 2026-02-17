from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = "data/noshow.csv"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Target: "Yes" = no-show (1), "No" = showed up (0)
    df["target"] = (df["No-show"] == "Yes").astype(int)

    # Parse datetimes
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce", utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce", utc=True)

    # Feature engineering
    df["lead_time_hours"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / 3600.0
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour
    df["scheduled_weekday"] = df["ScheduledDay"].dt.day_name()
    df["appointment_weekday"] = df["AppointmentDay"].dt.day_name()

    # Fix known data issue
    df = df[df["Age"].between(0, 120)]

    # Drop identifiers / leakage-prone columns + raw target/datetimes
    drop_cols = ["PatientId", "AppointmentID", "No-show", "ScheduledDay", "AppointmentDay"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def build_model(X: pd.DataFrame) -> Pipeline:
    # Column types (include "string" to avoid pandas warning in newer versions)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def evaluate_at_threshold(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    preds = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def tune_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    target_recall: float = 0.70,
    n_thresholds: int = 101,
) -> tuple[pd.DataFrame, float, float]:
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    rows = [evaluate_at_threshold(y_true, proba, t) for t in thresholds]
    df = pd.DataFrame(rows)

    # Best threshold by F1
    best_f1_row = df.loc[df["f1"].idxmax()]
    best_f1_threshold = float(best_f1_row["threshold"])

    # Threshold achieving at least target recall with best precision (or closest if none)
    df_meet = df[df["recall"] >= target_recall]
    if len(df_meet) > 0:
        best_recall_threshold = float(df_meet.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]["threshold"])
    else:
        # If target recall is impossible, choose the closest recall
        best_recall_threshold = float(df.iloc[(df["recall"] - target_recall).abs().idxmin()]["threshold"])

    return df, best_f1_threshold, best_recall_threshold


def print_eval(y_true: np.ndarray, proba: np.ndarray, threshold: float, title: str) -> None:
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    print(f"\n=== {title} ===")
    print(f"Threshold: {threshold:.2f}")
    print("Confusion matrix (TN FP / FN TP):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, preds, digits=4))


def save_artifacts(
    model: Pipeline,
    best_f1_threshold: float,
    best_recall_threshold: float,
    roc_auc: float,
    pr_auc: float,
    best_f1_row: pd.Series,
    best_recall_row: pd.Series,
) -> None:
    """Persist model, threshold, and training metrics to artifacts/."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # model.joblib
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"\nSaved model        → {model_path}")

    # threshold.json
    threshold_data = {
        "best_f1_threshold": best_f1_threshold,
        "best_recall_threshold": best_recall_threshold,
        "deployment_threshold": best_recall_threshold,
        "note": "deployment_threshold defaults to best_recall_threshold (recall >= 0.70)",
    }
    threshold_path = os.path.join(ARTIFACTS_DIR, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved thresholds   → {threshold_path}")

    # training_metrics.json
    metrics_data = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "best_f1_threshold": {
            "threshold": round(best_f1_threshold, 4),
            "precision": round(float(best_f1_row["precision"]), 4),
            "recall": round(float(best_f1_row["recall"]), 4),
            "f1": round(float(best_f1_row["f1"]), 4),
        },
        "best_recall_threshold": {
            "threshold": round(best_recall_threshold, 4),
            "precision": round(float(best_recall_row["precision"]), 4),
            "recall": round(float(best_recall_row["recall"]), 4),
            "f1": round(float(best_recall_row["f1"]), 4),
        },
    }
    metrics_path = os.path.join(ARTIFACTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics      → {metrics_path}")


def main() -> None:
    data_path = os.path.join(PROJECT_ROOT, DATA_PATH)
    df = load_and_clean(data_path)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(X_train)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print("=== Evaluation (Threshold-independent) ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    # Tune thresholds
    metrics_df, best_f1_t, best_recall_t = tune_thresholds(y_test, proba, target_recall=0.70)

    # Show a quick summary of the best points
    best_f1_row = metrics_df.loc[metrics_df["f1"].idxmax()]
    print("\n=== Best threshold by F1 ===")
    print(best_f1_row.to_string(index=False))

    # Find the row used for target recall threshold (closest match)
    best_recall_row = metrics_df.iloc[(metrics_df["threshold"] - best_recall_t).abs().idxmin()]
    print("\n=== Threshold targeting recall >= 0.70 (best precision among those) ===")
    print(best_recall_row.to_string(index=False))

    # Print detailed confusion matrices / reports for both thresholds
    print_eval(y_test, proba, best_f1_t, "Evaluation at Best-F1 Threshold")
    print_eval(y_test, proba, best_recall_t, "Evaluation at Target-Recall Threshold (>= 0.70)")

    # Save all artifacts
    save_artifacts(model, best_f1_t, best_recall_t, roc_auc, pr_auc, best_f1_row, best_recall_row)

    print("\nTraining complete. Artifacts saved to artifacts/")


if __name__ == "__main__":
    main()
