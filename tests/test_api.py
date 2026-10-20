from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_score_endpoint_returns_valid_response():
    payload = {
        "amount": 250.0,
        "account_age_days": 365.0,
        "transactions_last_hour": 2.0,
        "distance_from_usual_km": 5.0,
        "failed_attempts": 0.0,
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] in {"high_risk", "low_risk"}
    assert "threshold" in data
    assert "model_dataset" in data


def test_score_endpoint_rejects_negative_amount():
    payload = {
        "amount": -1.0,
        "account_age_days": 365.0,
        "transactions_last_hour": 2.0,
        "distance_from_usual_km": 5.0,
        "failed_attempts": 0.0,
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 422
