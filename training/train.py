
from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"

FEATURE_NAMES = [
    "amount",
    "account_age_days",
    "transactions_last_hour",
    "distance_from_usual_km",
    "failed_attempts",
]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Temporary synthetic data for validating our training workflow.
    X, y = make_classification(
        n_samples=5000,
        n_features=len(FEATURE_NAMES),
        n_informative=4,
        n_redundant=0,
        weights=[0.95, 0.05],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "dataset": "synthetic-development-only",
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
