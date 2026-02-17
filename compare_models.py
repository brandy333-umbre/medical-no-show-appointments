"""
PHASE 3 — Model Comparison & Robustness
========================================
Compares Logistic Regression (baseline), Random Forest, and Gradient Boosting.

Evaluates:
  - ROC-AUC and PR-AUC via Stratified K-Fold cross-validation
  - Performance variance (stability)
  - Calibration quality (Brier score + calibration curves)
  - Overfitting check (train vs. test gap)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import sys
import os

# Ensure src/ is on the path so train_model can be imported from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import load_and_clean, DATA_PATH

# Resolve DATA_PATH relative to project root (one level above src/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESOLVED_DATA_PATH = os.path.join(PROJECT_ROOT, DATA_PATH)

warnings.filterwarnings("ignore", category=UserWarning)

N_FOLDS = 5
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Preprocessing (mirrors train_model.py but built fresh per fold)
# ------------------------------------------------------------------
def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string", "bool"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------
def get_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }


def _build_pipeline(preprocessor: ColumnTransformer, clf: object) -> Pipeline:
    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])


# ------------------------------------------------------------------
# Cross-validation loop
# ------------------------------------------------------------------
def cross_validate_models(
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """Run Stratified K-Fold CV and collect per-fold metrics for each model."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, clf in models.items():
            preprocessor = _build_preprocessor(X_train)
            pipe = _build_pipeline(preprocessor, clf)
            pipe.fit(X_train, y_train)

            proba_test = pipe.predict_proba(X_test)[:, 1]
            proba_train = pipe.predict_proba(X_train)[:, 1]

            roc_test = roc_auc_score(y_test, proba_test)
            pr_test = average_precision_score(y_test, proba_test)
            brier_test = brier_score_loss(y_test, proba_test)

            roc_train = roc_auc_score(y_train, proba_train)
            pr_train = average_precision_score(y_train, proba_train)

            records.append({
                "model": name,
                "fold": fold_idx,
                "roc_auc_test": roc_test,
                "pr_auc_test": pr_test,
                "brier_test": brier_test,
                "roc_auc_train": roc_train,
                "pr_auc_train": pr_train,
            })

            print(f"  Fold {fold_idx} | {name:<25s} | "
                  f"ROC-AUC {roc_test:.4f} | PR-AUC {pr_test:.4f} | "
                  f"Brier {brier_test:.4f}")

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# Reporting helpers
# ------------------------------------------------------------------
def print_summary_table(results: pd.DataFrame) -> None:
    """Aggregate per-model means and standard deviations."""
    print("\n" + "=" * 72)
    print("MODEL COMPARISON SUMMARY  (Stratified {}-Fold CV)".format(N_FOLDS))
    print("=" * 72)

    agg = results.groupby("model").agg(
        roc_auc_mean=("roc_auc_test", "mean"),
        roc_auc_std=("roc_auc_test", "std"),
        pr_auc_mean=("pr_auc_test", "mean"),
        pr_auc_std=("pr_auc_test", "std"),
        brier_mean=("brier_test", "mean"),
        brier_std=("brier_test", "std"),
    )

    for name, row in agg.iterrows():
        print(f"\n--- {name} ---")
        print(f"  ROC-AUC : {row.roc_auc_mean:.4f} ± {row.roc_auc_std:.4f}")
        print(f"  PR-AUC  : {row.pr_auc_mean:.4f} ± {row.pr_auc_std:.4f}")
        print(f"  Brier   : {row.brier_mean:.4f} ± {row.brier_std:.4f}  (lower is better)")


def print_stability_report(results: pd.DataFrame) -> None:
    """Report variance / coefficient of variation to demonstrate stability."""
    print("\n" + "=" * 72)
    print("STABILITY REPORT  (Performance Variance Across Folds)")
    print("=" * 72)

    for name, grp in results.groupby("model"):
        roc_cv = grp["roc_auc_test"].std() / grp["roc_auc_test"].mean() * 100
        pr_cv = grp["pr_auc_test"].std() / grp["pr_auc_test"].mean() * 100
        print(f"\n--- {name} ---")
        print(f"  ROC-AUC CV%: {roc_cv:.2f}%   (std / mean × 100)")
        print(f"  PR-AUC  CV%: {pr_cv:.2f}%")
        if roc_cv < 2.0:
            print("  → Very stable across folds")
        elif roc_cv < 5.0:
            print("  → Reasonably stable")
        else:
            print("  → High variance — consider more data or regularisation")


def print_overfitting_check(results: pd.DataFrame) -> None:
    """Compare train vs test metrics to flag overfitting."""
    print("\n" + "=" * 72)
    print("OVERFITTING CHECK  (Train vs. Test Gap)")
    print("=" * 72)

    agg = results.groupby("model").agg(
        roc_train=("roc_auc_train", "mean"),
        roc_test=("roc_auc_test", "mean"),
        pr_train=("pr_auc_train", "mean"),
        pr_test=("pr_auc_test", "mean"),
    )

    for name, row in agg.iterrows():
        roc_gap = row.roc_train - row.roc_test
        pr_gap = row.pr_train - row.pr_test
        print(f"\n--- {name} ---")
        print(f"  ROC-AUC  train: {row.roc_train:.4f}  test: {row.roc_test:.4f}  gap: {roc_gap:+.4f}")
        print(f"  PR-AUC   train: {row.pr_train:.4f}  test: {row.pr_test:.4f}  gap: {pr_gap:+.4f}")
        if roc_gap < 0.02:
            print("  → Minimal overfitting")
        elif roc_gap < 0.05:
            print("  → Moderate overfitting — monitor closely")
        else:
            print("  → Significant overfitting — consider stronger regularisation")


def print_calibration_report(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Fit each model once on a held-out split and print calibration curves."""
    print("\n" + "=" * 72)
    print("CALIBRATION ANALYSIS  (10-bin reliability diagram)")
    print("=" * 72)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = get_models()
    for name, clf in models.items():
        preprocessor = _build_preprocessor(X_train)
        pipe = _build_pipeline(preprocessor, clf)
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, proba)

        fraction_pos, mean_predicted = calibration_curve(
            y_test, proba, n_bins=10, strategy="uniform"
        )

        print(f"\n--- {name}  (Brier score: {brier:.4f}) ---")
        print(f"  {'Bin':>4s}  {'Mean predicted':>15s}  {'Fraction positive':>18s}  {'Deviation':>10s}")
        for i, (mp, fp) in enumerate(zip(mean_predicted, fraction_pos), start=1):
            dev = fp - mp
            print(f"  {i:>4d}  {mp:>15.4f}  {fp:>18.4f}  {dev:>+10.4f}")

        # Overall assessment
        max_dev = np.max(np.abs(fraction_pos - mean_predicted))
        if max_dev < 0.05:
            print("  → Well calibrated")
        elif max_dev < 0.10:
            print("  → Reasonably calibrated")
        else:
            print("  → Poor calibration — consider Platt scaling or isotonic regression")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    print("Loading and cleaning data …")
    df = load_and_clean(RESOLVED_DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    print(f"Dataset: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"No-show rate: {y.mean():.2%}\n")

    # --- Stratified K-Fold CV ---
    print(f"Running {N_FOLDS}-fold Stratified Cross-Validation …\n")
    results = cross_validate_models(X, y)

    # --- Reports ---
    print_summary_table(results)
    print_stability_report(results)
    print_overfitting_check(results)

    # --- Calibration (using first fold split for demonstration) ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    first_train_idx, first_test_idx = next(iter(skf.split(X, y)))
    print_calibration_report(X, y, first_train_idx, first_test_idx)

    # --- Final recommendation ---
    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    agg = results.groupby("model")["roc_auc_test"].mean()
    best_model = agg.idxmax()
    print(f"\nBest model by mean ROC-AUC: {best_model} ({agg[best_model]:.4f})")

    # Check if improvement over baseline is meaningful
    lr_auc = agg.get("Logistic Regression", 0)
    best_auc = agg[best_model]
    if best_model != "Logistic Regression" and (best_auc - lr_auc) < 0.005:
        print("  → Improvement over Logistic Regression is marginal (<0.5%).")
        print("  → Logistic Regression may still be preferred for interpretability.")
    elif best_model == "Logistic Regression":
        print("  → Baseline model is already the strongest — no need for complexity.")
    else:
        improvement = (best_auc - lr_auc) * 100
        print(f"  → Improvement of {improvement:.2f}% ROC-AUC over Logistic Regression.")
        print(f"  → Consider deploying {best_model} if interpretability is not critical.")

    print()


if __name__ == "__main__":
    main()
