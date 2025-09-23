from fastapi.testclient import TestClient
from app import app

def test_api_predict_single(sample_payload):
    client = TestClient(app)
    resp = client.post("/predict?threshold=0.5", json=sample_payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "churn_probability" in data and "churn_pred" in data