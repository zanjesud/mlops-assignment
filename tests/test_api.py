from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check_with_model():
    """Test health check endpoint when model is loaded"""
    with patch("api.main.model", MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["model_loaded"] is True


def test_health_check_without_model():
    """Test health check endpoint when model is not loaded"""
    with patch("api.main.model", None):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "timestamp" in data
        assert data["model_loaded"] is False


def test_root_endpoint_with_model():
    """Test root endpoint when model is loaded"""
    with patch("api.main.model", MagicMock()):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Iris Classifier API"
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/docs"
        assert data["metrics"] == "/metrics"
        assert data["model_loaded"] is True


def test_root_endpoint_without_model():
    """Test root endpoint when model is not loaded"""
    with patch("api.main.model", None):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False


@patch("api.main.conn")
@patch("api.main.model")
def test_predict_success_setosa(mock_model, mock_conn):
    """Test successful prediction for setosa"""
    mock_model.predict.return_value = np.array([0])
    mock_model.__bool__ = lambda x: True

    response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["predictions"] == [0]
    mock_conn.execute.assert_called()
    mock_conn.commit.assert_called()


@patch("api.main.conn")
@patch("api.main.model")
def test_predict_success_versicolor(mock_model, mock_conn):
    """Test successful prediction for versicolor"""
    mock_model.predict.return_value = np.array([1])
    mock_model.__bool__ = lambda x: True

    response = client.post("/predict", json={"data": [[6.0, 2.9, 4.5, 1.5]]})
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [1]


@patch("api.main.conn")
@patch("api.main.model")
def test_predict_success_virginica(mock_model, mock_conn):
    """Test successful prediction for virginica"""
    mock_model.predict.return_value = np.array([2])
    mock_model.__bool__ = lambda x: True

    response = client.post("/predict", json={"data": [[6.3, 3.3, 6.0, 2.5]]})
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [2]


@patch("api.main.conn")
@patch("api.main.model")
def test_predict_multiple_predictions(mock_model, mock_conn):
    """Test multiple predictions in one request"""
    mock_model.predict.return_value = np.array([0, 1, 2])
    mock_model.__bool__ = lambda x: True

    response = client.post(
        "/predict",
        json={
            "data": [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5], [6.3, 3.3, 6.0, 2.5]]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == [0, 1, 2]


def test_predict_no_model():
    """Test prediction when model is not loaded"""
    with patch("api.main.model", None):
        response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"] == "Model not available"


@patch("api.main.model")
def test_predict_model_error(mock_model):
    """Test prediction when model throws an error"""
    mock_model.predict.side_effect = Exception("Model prediction failed")
    mock_model.__bool__ = lambda x: True

    response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "Prediction failed: Model prediction failed" in data["error"]


@patch("api.main.model")
def test_predict_dataframe_error(mock_model):
    """Test prediction when DataFrame creation fails"""
    mock_model.__bool__ = lambda x: True

    with patch("pandas.DataFrame", side_effect=Exception("DataFrame error")):
        response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "Prediction failed" in data["error"]


def test_custom_metrics_with_data():
    """Test custom metrics endpoint with prediction data"""
    with (
        patch("api.main.prediction_counter", 10),
        patch(
            "api.main.species_predictions",
            {"setosa": 3, "versicolor": 4, "virginica": 3},
        ),
        patch("api.main.model", MagicMock()),
    ):

        response = client.get("/custom-metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 10
        assert data["species_breakdown"] == {
            "setosa": 3,
            "versicolor": 4,
            "virginica": 3,
        }
        assert data["model_status"] == "loaded"
        assert data["database_status"] == "connected"


def test_custom_metrics_no_model():
    """Test custom metrics endpoint when model is not loaded"""
    with patch("api.main.model", None):
        response = client.get("/custom-metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["model_status"] == "not_loaded"


def test_invalid_prediction_data():
    """Test prediction with invalid data format"""
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422  # Validation error


def test_prediction_with_unknown_species():
    """Test prediction with unknown species value"""
    with patch("api.main.conn"), patch("api.main.model") as mock_model:

        mock_model.predict.return_value = np.array([99])  # Unknown species
        mock_model.__bool__ = lambda x: True

        response = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"] == [99]


@patch("logging.info")
@patch("mlflow.pyfunc.load_model")
def test_model_loading_success(mock_load_model, mock_logging_info):
    """Test successful model loading at startup"""
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    # Reload the module to test startup code
    import importlib

    import api.main

    importlib.reload(api.main)

    mock_logging_info.assert_called_with("Model loaded successfully at startup")


@patch("logging.error")
@patch("mlflow.pyfunc.load_model")
def test_model_loading_failure(mock_load_model, mock_logging_error):
    """Test model loading failure at startup"""
    mock_load_model.side_effect = Exception("Model loading failed")

    # Reload the module to test startup code
    import importlib

    import api.main

    importlib.reload(api.main)

    mock_logging_error.assert_called()


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_docs_endpoint():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200
