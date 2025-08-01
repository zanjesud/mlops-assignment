import os

import pytest
import requests

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
UI_URL = os.getenv("UI_URL", "http://localhost:8501")


def test_api_health():
    """Test API health endpoint"""
    response = requests.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_prediction():
    """Test API prediction functionality"""
    test_data = {"data": [[5.1, 3.5, 1.4, 0.2]]}
    response = requests.post(f"{API_URL}/predict", json=test_data, timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1


def test_api_metrics():
    """Test API metrics endpoint"""
    response = requests.get(f"{API_URL}/metrics", timeout=5)
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_ui_health():
    """Test UI health endpoint"""
    response = requests.get(f"{UI_URL}/_stcore/health", timeout=5)
    assert response.status_code == 200


def test_end_to_end_prediction():
    """Test complete end-to-end prediction flow"""
    # First, make a prediction via API
    test_data = {"data": [[6.3, 3.3, 4.7, 1.6]]}
    response = requests.post(f"{API_URL}/predict", json=test_data, timeout=5)
    assert response.status_code == 200

    # Then check if UI is accessible
    response = requests.get(f"{UI_URL}", timeout=5)
    assert response.status_code == 200


def test_api_documentation():
    """Test API documentation accessibility"""
    response = requests.get(f"{API_URL}/docs", timeout=5)
    assert response.status_code == 200


def test_api_openapi():
    """Test OpenAPI schema endpoint"""
    response = requests.get(f"{API_URL}/openapi.json", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data


def test_multiple_predictions():
    """Test multiple predictions to check for consistency"""
    test_data = {
        "data": [[5.1, 3.5, 1.4, 0.2], [6.3, 3.3, 4.7, 1.6], [7.0, 3.2, 4.7, 1.4]]
    }
    response = requests.post(f"{API_URL}/predict", json=test_data, timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3


def test_error_handling():
    """Test error handling for invalid requests"""
    # Test with invalid data
    test_data = {"data": [[5.1, 3.5, 1.4]]}  # Missing one feature
    response = requests.post(f"{API_URL}/predict", json=test_data, timeout=5)
    assert response.status_code == 422


def test_service_dependencies():
    """Test that all required services are running"""
    services = [(f"{API_URL}/health", "API"), (f"{UI_URL}/_stcore/health", "UI")]

    for url, service_name in services:
        try:
            response = requests.get(url, timeout=5)
            assert response.status_code == 200, f"{service_name} is not healthy"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"{service_name} is not accessible: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
