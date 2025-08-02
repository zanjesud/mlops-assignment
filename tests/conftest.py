import os
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Create logs directory
    os.makedirs("logs", exist_ok=True)


@pytest.fixture
def mock_model():
    """Mock MLflow model for testing"""
    mock = Mock()
    mock.predict.return_value = np.array([0])  # Always predict class 0
    return mock


@pytest.fixture
def mock_mlflow_load_model(mock_model):
    """Mock mlflow.pyfunc.load_model"""
    with patch("api.main.mlflow.pyfunc.load_model", return_value=mock_model):
        yield mock_model
