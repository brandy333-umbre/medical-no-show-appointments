from __future__ import annotations

import pandas as pd
import numpy as np

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
)

DATA_PATH = "data/noshow.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    # Target: "Yes" = no-show (1), "No" = showed up (0)
    df["target"] = (df["No-show"] == "Yes").astype(int)

    # Parse datetimes (ScheduledDay includes time; AppointmentDay is date-like)
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], errors="coerce", utc=True)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce", utc=True)

    # Feature engineering (simple, high signal, explainable)
    # Lead time in hours (how far in advance scheduled)
    df["lead_time_hours"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.total_seconds() / 3600.0

    # Scheduled hour & weekday (often correlates with attendance)
    df["scheduled_hour"] = df["ScheduledDay"].dt.hour
    df["scheduled_weekday"] = df["ScheduledDay"].dt.day_name()

    # Appointment weekday (day-of-week effects)
    df["appointment_weekday"] = df["AppointmentDay"].dt.day_name()

    # Known data issue: negative ages exist in this dataset
    df = df[df["Age"].between(0, 120)]

    # Drop identifiers / leakage-prone columns
    drop_cols = [
        "PatientId",
        "AppointmentID",
        "No-show",
        "ScheduledDay",
        "AppointmentDay",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def main() -> None:
    df = load_and_clean(DATA_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    # Train/test split with stratification (important for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()


    # Preprocessing
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

    # Model: logistic regression baseline (fast, explainable, strong for tabular)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps if No-show is minority
        solver="liblinear",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    model.fit(X_train, y_train)

    # Probabilities for AUC / PR-AUC
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    # Choose a simple threshold for now
    threshold = 0.5
    preds = (proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, preds)

    print("=== Evaluation (Test Set) ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nConfusion matrix (threshold=0.5):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    # Optional: show the most influential features (logistic regression coefficients)
    # NOTE: after one-hot encoding, feature names expand.
    try:
        ohe = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)

        feature_names = np.concatenate([np.array(num_cols), cat_feature_names])
        coefs = model.named_steps["clf"].coef_[0]

        top_pos_idx = np.argsort(coefs)[-10:][::-1]
        top_neg_idx = np.argsort(coefs)[:10]

        print("\n=== Top Features Increasing No-Show Risk (positive coef) ===")
        for i in top_pos_idx:
            print(f"{feature_names[i]}: {coefs[i]:.4f}")

        print("\n=== Top Features Decreasing No-Show Risk (negative coef) ===")
        for i in top_neg_idx:
            print(f"{feature_names[i]}: {coefs[i]:.4f}")

    except Exception as e:
        print("\n(Could not print feature importances:", e, ")")


if __name__ == "__main__":
    main()
