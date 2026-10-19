from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Locate the project directory and saved model.
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

# Load the trained model once when the application starts.
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_names = artifact["feature_names"]
threshold = artifact["threshold"]

app = FastAPI(
    title="Transaction Risk Scoring Service",
    version="0.1.0",
    description="Development-only ML inference API",
)


class Transaction(BaseModel):
    amount: float = Field(ge=0)
    account_age_days: float = Field(ge=0)
    transactions_last_hour: float = Field(ge=0)
    distance_from_usual_km: float = Field(ge=0)
    failed_attempts: float = Field(ge=0)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/score")
def score(transaction: Transaction):
    features = np.array(
        [[getattr(transaction, name) for name in feature_names]],
        dtype=float,
    )

    probability = float(model.predict_proba(features)[0, 1])

    return {
        "risk_score": round(probability, 4),
        "threshold": threshold,
        "risk_label": (
            "high_risk" if probability >= threshold else "low_risk"
        ),
        "model_dataset": artifact["dataset"],
    }
