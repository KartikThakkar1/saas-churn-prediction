# tests/test_api.py
"""
API smoke tests.

- Uses TestClient **as a context manager** so FastAPI lifespan
  (startup/shutdown) runs and the Predictor loads at startup.
- Sends one valid payload to /predict and verifies the response shape.
"""

from fastapi.testclient import TestClient
from app import app

def test_api_predict_single(sample_payload):
    # Context manager triggers app startup (loads Predictor) and shutdown
    with TestClient(app) as client:
        # Health should be ok after startup
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] in {"ok", "booting"}  # allow "booting" during init on slow runners

        # Single prediction
        resp = client.post("/predict?threshold=0.5", json=sample_payload)
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert "churn_probability" in data
        assert "churn_pred" in data
        assert 0.0 <= float(data["churn_probability"]) <= 1.0
        assert int(data["churn_pred"]) in (0, 1)
