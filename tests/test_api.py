from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Iris Classifier API"
    assert data["version"] == "1.0.0"


def test_predict_endpoint():
    """Test prediction endpoint"""
    test_data = {"data": [[5.1, 3.5, 1.4, 0.2]]}
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert isinstance(data["predictions"][0], int)


def test_predict_invalid_data():
    """Test prediction with invalid data"""
    test_data = {"data": [[5.1, 3.5, 1.4]]}  # Missing one feature
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_docs_endpoint():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200
