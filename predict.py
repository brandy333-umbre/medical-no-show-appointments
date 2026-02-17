"""
PHASE 4 — Inference Component
==============================
Loads the trained model and optimised threshold from artifacts/.
Provides:
  - Single-patient prediction
  - Batch prediction
  - FastAPI REST endpoint (/predict and /predict/batch)

Output format per patient:
  {
    "no_show_probability": 0.72,
    "flag_for_intervention": true
  }

Run the API server:
    uvicorn src.predict:app --reload
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import load_and_clean


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------
def load_model():
    path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_model.py first."
        )
    return joblib.load(path)


def load_threshold() -> float:
    path = os.path.join(ARTIFACTS_DIR, "threshold.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Threshold file not found at {path}. Run train_model.py first."
        )
    with open(path) as f:
        data = json.load(f)
    return float(data["deployment_threshold"])


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
def predict_single(patient: dict, model=None, threshold: float | None = None) -> dict:
    """Return no-show probability and intervention flag for one patient."""
    if model is None:
        model = load_model()
    if threshold is None:
        threshold = load_threshold()

    df = pd.DataFrame([patient])
    proba = float(model.predict_proba(df)[:, 1][0])
    return {
        "no_show_probability": round(proba, 4),
        "flag_for_intervention": proba >= threshold,
    }


def predict_batch(
    patients: list[dict],
    model=None,
    threshold: float | None = None,
) -> list[dict]:
    """Return predictions for a list of patients."""
    if model is None:
        model = load_model()
    if threshold is None:
        threshold = load_threshold()

    df = pd.DataFrame(patients)
    probas = model.predict_proba(df)[:, 1]
    return [
        {
            "no_show_probability": round(float(p), 4),
            "flag_for_intervention": bool(p >= threshold),
        }
        for p in probas
    ]


# ---------------------------------------------------------------------------
# FastAPI app (lazy — only built when create_app() is called or uvicorn loads)
# ---------------------------------------------------------------------------
def create_app():
    """Build and return the FastAPI application.

    Imports fastapi/pydantic here so the CLI demo works without them installed.
    Install with:  pip install fastapi uvicorn
    """
    from pydantic import BaseModel, Field
    from fastapi import FastAPI, HTTPException

    class PatientInput(BaseModel):
        Gender: str = Field(..., example="F")
        Age: int = Field(..., example=62)
        Neighbourhood: str = Field(..., example="JARDIM CAMBURI")
        Scholarship: int = Field(..., example=0)
        Hipertension: int = Field(..., example=1)
        Diabetes: int = Field(..., example=0)
        Alcoholism: int = Field(..., example=0)
        Handcap: int = Field(..., example=0)
        SMS_received: int = Field(..., example=1)
        lead_time_hours: float = Field(..., example=48.0)
        scheduled_hour: int = Field(..., example=9)
        scheduled_weekday: str = Field(..., example="Monday")
        appointment_weekday: str = Field(..., example="Wednesday")

    class PredictionOutput(BaseModel):
        no_show_probability: float
        flag_for_intervention: bool

    class BatchInput(BaseModel):
        patients: list[PatientInput]

    class BatchOutput(BaseModel):
        predictions: list[PredictionOutput]

    application = FastAPI(
        title="No-Show Appointment Prediction API",
        description="Predicts whether a patient will miss a scheduled medical appointment.",
        version="1.0.0",
    )

    _state = {"model": None, "threshold": None}

    @application.on_event("startup")
    def startup_load_artifacts():
        try:
            _state["model"] = load_model()
            _state["threshold"] = load_threshold()
            print(f"Model loaded. Deployment threshold: {_state['threshold']:.4f}")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("The API will return errors until artifacts are available.")

    @application.get("/health")
    def health_check():
        return {
            "status": "healthy" if _state["model"] is not None else "model_not_loaded",
            "deployment_threshold": _state["threshold"],
        }

    @application.post("/predict", response_model=PredictionOutput)
    def predict_endpoint(patient: PatientInput):
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
        result = predict_single(patient.model_dump(), model=_state["model"], threshold=_state["threshold"])
        return result

    @application.post("/predict/batch", response_model=BatchOutput)
    def predict_batch_endpoint(batch: BatchInput):
        if _state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
        patients_dicts = [p.model_dump() for p in batch.patients]
        results = predict_batch(patients_dicts, model=_state["model"], threshold=_state["threshold"])
        return {"predictions": results}

    return application


# Module-level app object for uvicorn (e.g. uvicorn src.predict:app)
# Only created when fastapi is available; silently skipped for CLI usage.
try:
    app = create_app()
except ImportError:
    app = None


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def demo() -> None:
    """Quick command-line demo with sample patients."""
    print("=" * 60)
    print("PREDICTION DEMO")
    print("=" * 60)

    model = load_model()
    threshold = load_threshold()
    print(f"Deployment threshold: {threshold:.4f}\n")

    sample_patients = [
        {
            "Gender": "F",
            "Age": 62,
            "Neighbourhood": "JARDIM CAMBURI",
            "Scholarship": 0,
            "Hipertension": 1,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 1,
            "lead_time_hours": 48.0,
            "scheduled_hour": 9,
            "scheduled_weekday": "Monday",
            "appointment_weekday": "Wednesday",
        },
        {
            "Gender": "M",
            "Age": 22,
            "Neighbourhood": "MARIA ORTIZ",
            "Scholarship": 1,
            "Hipertension": 0,
            "Diabetes": 0,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 0,
            "lead_time_hours": 336.0,
            "scheduled_hour": 14,
            "scheduled_weekday": "Friday",
            "appointment_weekday": "Tuesday",
        },
        {
            "Gender": "F",
            "Age": 45,
            "Neighbourhood": "RESISTENCIA",
            "Scholarship": 0,
            "Hipertension": 0,
            "Diabetes": 1,
            "Alcoholism": 0,
            "Handcap": 0,
            "SMS_received": 1,
            "lead_time_hours": 2.0,
            "scheduled_hour": 11,
            "scheduled_weekday": "Wednesday",
            "appointment_weekday": "Wednesday",
        },
    ]

    print("--- Single predictions ---\n")
    for i, patient in enumerate(sample_patients, 1):
        result = predict_single(patient, model=model, threshold=threshold)
        print(f"Patient {i}: Age={patient['Age']}, Gender={patient['Gender']}, "
              f"LeadTime={patient['lead_time_hours']}h")
        print(f"  → no_show_probability: {result['no_show_probability']:.4f}")
        print(f"  → flag_for_intervention: {result['flag_for_intervention']}\n")

    print("--- Batch prediction ---\n")
    batch_results = predict_batch(sample_patients, model=model, threshold=threshold)
    for i, res in enumerate(batch_results, 1):
        print(f"  Patient {i}: probability={res['no_show_probability']:.4f}, "
              f"intervene={res['flag_for_intervention']}")

    print()


if __name__ == "__main__":
    demo()
