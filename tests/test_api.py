from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_returns_status():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_file_exists" in payload


def test_metrics_returns_request_count():
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "request_count" in payload
    assert "avg_latency_ms" in payload
