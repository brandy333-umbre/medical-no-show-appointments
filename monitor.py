"""
PHASE 4 — Monitoring & Drift Detection
========================================
Simulates a production monitoring system that tracks:

1. Data Drift    — statistical shifts in feature distributions
2. Performance   — ROC-AUC, recall, precision at deployment threshold
3. Fairness      — performance parity across demographic groups

Usage:
    python src/monitor.py

The script splits historical data into a "reference" window (training period)
and a "production" window (simulated new data) to demonstrate monitoring logic.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import load_and_clean, DATA_PATH
from predict import load_model, load_threshold

warnings.filterwarnings("ignore", category=UserWarning)

# Thresholds for alerting
DRIFT_P_VALUE = 0.05
PERFORMANCE_FLOOR_ROC = 0.65
PERFORMANCE_FLOOR_RECALL = 0.60


# ===================================================================
# 1. DATA DRIFT MONITORING
# ===================================================================
def detect_numeric_drift(
    ref: pd.Series,
    prod: pd.Series,
    feature_name: str,
    p_threshold: float = DRIFT_P_VALUE,
) -> dict:
    """Kolmogorov-Smirnov test for numeric feature drift."""
    stat, p_value = stats.ks_2samp(ref.dropna(), prod.dropna())
    drifted = p_value < p_threshold
    return {
        "feature": feature_name,
        "test": "Kolmogorov-Smirnov",
        "statistic": round(stat, 4),
        "p_value": round(p_value, 6),
        "drifted": drifted,
        "ref_mean": round(float(ref.mean()), 4),
        "ref_std": round(float(ref.std()), 4),
        "prod_mean": round(float(prod.mean()), 4),
        "prod_std": round(float(prod.std()), 4),
    }


def detect_categorical_drift(
    ref: pd.Series,
    prod: pd.Series,
    feature_name: str,
    p_threshold: float = DRIFT_P_VALUE,
) -> dict:
    """Chi-squared test for categorical feature drift."""
    all_cats = set(ref.dropna().unique()) | set(prod.dropna().unique())

    ref_counts = ref.value_counts()
    prod_counts = prod.value_counts()

    ref_freq = np.array([ref_counts.get(c, 0) for c in sorted(all_cats)], dtype=float)
    prod_freq = np.array([prod_counts.get(c, 0) for c in sorted(all_cats)], dtype=float)

    # Normalise to expected counts (scale prod to ref total)
    if prod_freq.sum() == 0 or ref_freq.sum() == 0:
        return {
            "feature": feature_name,
            "test": "Chi-squared",
            "statistic": 0.0,
            "p_value": 1.0,
            "drifted": False,
        }

    expected = ref_freq / ref_freq.sum() * prod_freq.sum()
    # Avoid zero-expected bins
    mask = expected > 0
    if mask.sum() < 2:
        return {
            "feature": feature_name,
            "test": "Chi-squared",
            "statistic": 0.0,
            "p_value": 1.0,
            "drifted": False,
        }

    stat, p_value = stats.chisquare(prod_freq[mask], f_exp=expected[mask])
    drifted = p_value < p_threshold
    return {
        "feature": feature_name,
        "test": "Chi-squared",
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "drifted": drifted,
    }


def run_drift_monitoring(ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> list[dict]:
    """Check drift for key features specified in the manual."""
    results = []

    # Numeric features to monitor
    numeric_features = ["Age", "lead_time_hours"]
    for feat in numeric_features:
        if feat in ref_df.columns and feat in prod_df.columns:
            results.append(detect_numeric_drift(ref_df[feat], prod_df[feat], feat))

    # Categorical features to monitor
    cat_features = ["appointment_weekday", "scheduled_weekday"]
    for feat in cat_features:
        if feat in ref_df.columns and feat in prod_df.columns:
            results.append(detect_categorical_drift(ref_df[feat], prod_df[feat], feat))

    # No-show rate drift (proportion test)
    if "target" in ref_df.columns and "target" in prod_df.columns:
        ref_rate = ref_df["target"].mean()
        prod_rate = prod_df["target"].mean()
        # Two-proportion z-test
        n1, n2 = len(ref_df), len(prod_df)
        p_pool = (ref_rate * n1 + prod_rate * n2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool > 0 else 1
        z = (prod_rate - ref_rate) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        results.append({
            "feature": "no_show_rate",
            "test": "Two-proportion Z-test",
            "statistic": round(float(z), 4),
            "p_value": round(float(p_value), 6),
            "drifted": p_value < DRIFT_P_VALUE,
            "ref_rate": round(float(ref_rate), 4),
            "prod_rate": round(float(prod_rate), 4),
        })

    return results


def print_drift_report(drift_results: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("DATA DRIFT MONITORING REPORT")
    print("=" * 72)

    any_drift = False
    for r in drift_results:
        status = "DRIFT DETECTED" if r["drifted"] else "OK"
        symbol = "!!" if r["drifted"] else "  "
        print(f"\n{symbol} {r['feature']:<25s}  [{r['test']}]")
        print(f"     statistic: {r['statistic']:.4f}   p-value: {r['p_value']:.6f}   → {status}")

        if "ref_mean" in r:
            print(f"     ref: mean={r['ref_mean']:.4f}, std={r['ref_std']:.4f}")
            print(f"     prod: mean={r['prod_mean']:.4f}, std={r['prod_std']:.4f}")
        if "ref_rate" in r:
            print(f"     ref rate: {r['ref_rate']:.4f}   prod rate: {r['prod_rate']:.4f}")

        if r["drifted"]:
            any_drift = True

    print("\n" + "-" * 72)
    if any_drift:
        print("ACTION REQUIRED: Significant drift detected. Consider retraining.")
    else:
        print("All features within expected distributions. No action needed.")


# ===================================================================
# 2. PERFORMANCE MONITORING
# ===================================================================
def run_performance_monitoring(
    y_true: np.ndarray,
    probas: np.ndarray,
    threshold: float,
) -> dict:
    """Compute performance metrics at the deployment threshold."""
    from sklearn.metrics import (
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
    )

    preds = (probas >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, probas)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    return {
        "roc_auc": round(roc_auc, 4),
        "precision_at_threshold": round(precision, 4),
        "recall_at_threshold": round(recall, 4),
        "f1_at_threshold": round(f1, 4),
        "threshold": round(threshold, 4),
        "n_samples": len(y_true),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def print_performance_report(perf: dict, reference_metrics: dict | None = None) -> None:
    print("\n" + "=" * 72)
    print("PERFORMANCE MONITORING REPORT")
    print("=" * 72)
    print(f"\n  Deployment threshold: {perf['threshold']:.4f}")
    print(f"  Samples evaluated:   {perf['n_samples']:,}")
    print(f"  Positive rate:       {perf['positive_rate']:.4f}")
    print()
    print(f"  ROC-AUC:             {perf['roc_auc']:.4f}", end="")
    if reference_metrics and "roc_auc" in reference_metrics:
        delta = perf["roc_auc"] - reference_metrics["roc_auc"]
        print(f"  (training: {reference_metrics['roc_auc']:.4f}, delta: {delta:+.4f})")
    else:
        print()

    print(f"  Precision:           {perf['precision_at_threshold']:.4f}")
    print(f"  Recall:              {perf['recall_at_threshold']:.4f}")
    print(f"  F1:                  {perf['f1_at_threshold']:.4f}")

    # Retraining triggers
    print("\n" + "-" * 72)
    alerts = []
    if perf["roc_auc"] < PERFORMANCE_FLOOR_ROC:
        alerts.append(f"ROC-AUC ({perf['roc_auc']:.4f}) below floor ({PERFORMANCE_FLOOR_ROC})")
    if perf["recall_at_threshold"] < PERFORMANCE_FLOOR_RECALL:
        alerts.append(f"Recall ({perf['recall_at_threshold']:.4f}) below floor ({PERFORMANCE_FLOOR_RECALL})")

    if alerts:
        print("RETRAINING TRIGGERED:")
        for a in alerts:
            print(f"  !! {a}")
    else:
        print("Performance within acceptable bounds. No retraining needed.")


# ===================================================================
# 3. FAIRNESS MONITORING
# ===================================================================
def run_fairness_monitoring(
    df: pd.DataFrame,
    probas: np.ndarray,
    threshold: float,
    group_col: str,
    group_label: str,
    max_groups: int = 10,
) -> list[dict]:
    """Evaluate performance parity across groups of a given column."""
    from sklearn.metrics import recall_score, precision_score, f1_score

    preds = (probas >= threshold).astype(int)
    y_true = df["target"].to_numpy()

    groups = df[group_col].value_counts().head(max_groups).index.tolist()
    results = []

    for grp in groups:
        mask = df[group_col] == grp
        if mask.sum() < 30:
            continue
        y_g = y_true[mask]
        p_g = preds[mask]
        prob_g = probas[mask]

        n_total = int(mask.sum())
        n_pos = int(y_g.sum())
        rec = recall_score(y_g, p_g, zero_division=0)
        prec = precision_score(y_g, p_g, zero_division=0)
        f1 = f1_score(y_g, p_g, zero_division=0)

        results.append({
            "group": str(grp),
            "n": n_total,
            "n_positive": n_pos,
            "no_show_rate": round(n_pos / n_total, 4) if n_total else 0,
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
        })

    return results


def print_fairness_report(
    results: list[dict],
    group_label: str,
) -> None:
    print(f"\n  --- Fairness by {group_label} (top groups) ---")
    print(f"  {'Group':<25s}  {'N':>6s}  {'NoShow%':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'F1':>7s}")

    recalls = []
    for r in results:
        print(f"  {r['group']:<25s}  {r['n']:>6d}  {r['no_show_rate']:>7.2%}  "
              f"{r['recall']:>7.4f}  {r['precision']:>7.4f}  {r['f1']:>7.4f}")
        recalls.append(r["recall"])

    if len(recalls) >= 2:
        gap = max(recalls) - min(recalls)
        print(f"\n  Max recall gap across groups: {gap:.4f}")
        if gap < 0.10:
            print("  → Acceptable parity")
        elif gap < 0.20:
            print("  → Moderate disparity — monitor closely")
        else:
            print("  → Large disparity — investigate potential bias")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("=" * 72)
    print("PRODUCTION MONITORING SIMULATION")
    print("=" * 72)

    # Load data and split into reference (80%) and production (20%) windows
    data_path = os.path.join(PROJECT_ROOT, DATA_PATH)
    df = load_and_clean(data_path)

    n = len(df)
    split = int(n * 0.8)
    ref_df = df.iloc[:split].copy()
    prod_df = df.iloc[split:].copy()

    print(f"\nReference window:  {len(ref_df):,} records")
    print(f"Production window: {len(prod_df):,} records")

    # Load model and threshold
    model = load_model()
    threshold = load_threshold()
    print(f"Deployment threshold: {threshold:.4f}")

    # ------------------------------------------------------------------
    # 1. DATA DRIFT
    # ------------------------------------------------------------------
    drift_results = run_drift_monitoring(ref_df, prod_df)
    print_drift_report(drift_results)

    # ------------------------------------------------------------------
    # 2. PERFORMANCE MONITORING
    # ------------------------------------------------------------------
    X_prod = prod_df.drop(columns=["target"])
    y_prod = prod_df["target"].to_numpy()
    probas = model.predict_proba(X_prod)[:, 1]

    # Load reference metrics if available
    reference_metrics = None
    metrics_path = os.path.join(ARTIFACTS_DIR, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            reference_metrics = json.load(f)

    perf = run_performance_monitoring(y_prod, probas, threshold)
    print_performance_report(perf, reference_metrics)

    # ------------------------------------------------------------------
    # 3. FAIRNESS MONITORING
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FAIRNESS MONITORING REPORT")
    print("=" * 72)

    # By Neighbourhood (top 10)
    if "Neighbourhood" in prod_df.columns:
        neighbourhood_results = run_fairness_monitoring(
            prod_df, probas, threshold, "Neighbourhood", "Neighbourhood", max_groups=10
        )
        print_fairness_report(neighbourhood_results, "Neighbourhood")

    # By Age group
    prod_df_age = prod_df.copy()
    prod_df_age["age_group"] = pd.cut(
        prod_df_age["Age"],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["0-18", "19-35", "36-50", "51-65", "66+"],
    )
    age_results = run_fairness_monitoring(
        prod_df_age, probas, threshold, "age_group", "Age Group", max_groups=10
    )
    print_fairness_report(age_results, "Age Group")

    # By Scholarship (socioeconomic proxy)
    if "Scholarship" in prod_df.columns:
        prod_df_schol = prod_df.copy()
        prod_df_schol["scholarship_status"] = prod_df_schol["Scholarship"].map(
            {0: "No scholarship", 1: "Has scholarship"}
        )
        schol_results = run_fairness_monitoring(
            prod_df_schol, probas, threshold, "scholarship_status",
            "Scholarship (Socioeconomic Proxy)", max_groups=5,
        )
        print_fairness_report(schol_results, "Scholarship (Socioeconomic Proxy)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MONITORING COMPLETE")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
