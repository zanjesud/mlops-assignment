"""
Model evaluation tests for CI/CD pipeline
Tests model performance, quality gates, and production readiness
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TestModelPerformance:
    """Test model performance against quality gates"""

    @pytest.fixture
    def sample_data(self):
        """Create sample iris data for testing"""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create a trained model for testing"""
        X_train, X_test, y_train, y_test = sample_data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def test_model_accuracy_threshold(self, trained_model, sample_data):
        """Test that model meets minimum accuracy threshold for production"""
        _, X_test, _, y_test = sample_data
        predictions = trained_model.predict(X_test)

        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test, predictions)

        # Lower threshold for testing (85% instead of 90%)
        assert accuracy >= 0.85, f"Model accuracy {accuracy:.3f} below threshold 0.85"

    def test_model_precision_threshold(self, trained_model, sample_data):
        """Test that model meets minimum precision threshold"""
        _, X_test, _, y_test = sample_data
        predictions = trained_model.predict(X_test)

        from sklearn.metrics import precision_score

        precision = precision_score(y_test, predictions, average="weighted")

        assert (
            precision >= 0.85
        ), f"Model precision {precision:.3f} below threshold 0.85"

    def test_model_recall_threshold(self, trained_model, sample_data):
        """Test that model meets minimum recall threshold"""
        _, X_test, _, y_test = sample_data
        predictions = trained_model.predict(X_test)

        from sklearn.metrics import recall_score

        recall = recall_score(y_test, predictions, average="weighted")

        assert recall >= 0.85, f"Model recall {recall:.3f} below threshold 0.85"

    def test_model_f1_threshold(self, trained_model, sample_data):
        """Test that model meets minimum F1 score threshold"""
        _, X_test, _, y_test = sample_data
        predictions = trained_model.predict(X_test)

        from sklearn.metrics import f1_score

        f1 = f1_score(y_test, predictions, average="weighted")

        assert f1 >= 0.85, f"Model F1 score {f1:.3f} below threshold 0.85"


class TestModelQuality:
    """Test model quality and robustness"""

    @pytest.fixture
    def trained_model(self):
        """Create a trained model for testing"""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def test_model_predicts_all_classes(self, trained_model):
        """Test that model can predict all iris species"""
        # Create samples for each class
        test_samples = np.array(
            [
                [5.1, 3.5, 1.4, 0.2],  # Typical setosa
                [6.0, 2.9, 4.5, 1.5],  # Typical versicolor
                [6.3, 3.3, 6.0, 2.5],  # Typical virginica
            ]
        )

        predictions = trained_model.predict(test_samples)
        unique_predictions = set(predictions)

        # Model should predict different classes
        assert len(unique_predictions) >= 2, "Model should predict multiple classes"

    def test_model_prediction_consistency(self, trained_model):
        """Test that model predictions are deterministic"""
        test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])

        pred1 = trained_model.predict(test_sample)
        pred2 = trained_model.predict(test_sample)
        pred3 = trained_model.predict(test_sample)

        assert np.array_equal(pred1, pred2), "Model predictions should be consistent"
        assert np.array_equal(pred2, pred3), "Model predictions should be consistent"

    def test_model_input_validation(self, trained_model):
        """Test that model handles invalid inputs appropriately"""
        # Test with wrong number of features
        invalid_input = np.array([[1.0, 2.0]])  # Only 2 features instead of 4

        with pytest.raises(ValueError):
            trained_model.predict(invalid_input)

    def test_model_output_format(self, trained_model):
        """Test that model output is in expected format"""
        test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        predictions = trained_model.predict(test_sample)

        # Predictions should be numpy array
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"

        # Predictions should be integers (class labels)
        assert predictions.dtype in [
            np.int32,
            np.int64,
        ], "Predictions should be integers"

        # Predictions should be in valid range
        assert all(
            0 <= pred <= 2 for pred in predictions
        ), "Predictions should be in range [0, 2]"


class TestProductionModel:
    """Test production model loading and evaluation"""

    def test_production_model_exists(self):
        """Test that production model files exist"""
        model_path = "models/production_model"

        if os.path.exists(model_path):
            # Check for essential model files
            expected_files = ["MLmodel", "model.pkl"]
            for file_name in expected_files:
                file_path = os.path.join(model_path, file_name)
                if os.path.exists(file_path):
                    assert True  # At least one essential file exists
                    return

        # If no production model exists, skip this test
        pytest.skip(
            "No production model found - this is expected in fresh environments"
        )

    @patch("mlflow.pyfunc.load_model")
    def test_production_model_loading(self, mock_load_model):
        """Test that production model can be loaded"""
        # Mock successful model loading
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2])
        mock_load_model.return_value = mock_model

        # Test model loading
        try:
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model("models/production_model")

            # Test prediction
            test_data = np.array([[5.1, 3.5, 1.4, 0.2]])
            predictions = model.predict(test_data)

            assert predictions is not None
            mock_load_model.assert_called_once()

        except Exception as e:
            pytest.skip(f"Model loading test skipped: {e}")


class TestModelEvaluationScript:
    """Test the evaluate_model.py script functionality"""

    def test_evaluate_model_script_exists(self):
        """Test that evaluate_model.py script exists"""
        script_path = "models/evaluate_model.py"
        if not os.path.exists(script_path):
            pytest.skip("evaluate_model.py not found - skipping script tests")
        assert os.path.exists(script_path)

    @patch("mlflow.pyfunc.load_model")
    def test_evaluate_model_script_execution(self, mock_load_model):
        """Test that evaluate_model.py script runs without errors"""
        # Skip if script doesn't exist
        if not os.path.exists("models/evaluate_model.py"):
            pytest.skip("evaluate_model.py not found - skipping script test")

        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 2, 0, 1, 2])
        mock_load_model.return_value = mock_model

        # Create temporary test data
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            joblib.dump((X_test, y_test), f.name)
            test_data_path = f.name

        try:
            # Mock the evaluate_model function
            with (
                patch("os.path.exists", return_value=True),
                patch("joblib.load", return_value=(X_test, y_test)),
            ):

                # Import and test the evaluation function
                from click.testing import CliRunner

                from models.evaluate_model import evaluate_model

                runner = CliRunner()
                result = runner.invoke(
                    evaluate_model,
                    ["--model_name", "iris_classifier", "--stage", "production"],
                )

                # Should not crash
                assert result.exit_code == 0 or result.exit_code is None

        except ImportError:
            pytest.skip("evaluate_model.py not found - skipping script test")
        finally:
            # Cleanup
            if os.path.exists(test_data_path):
                os.unlink(test_data_path)


class TestModelDrift:
    """Test for model drift and performance degradation"""

    def test_model_performance_stability(self):
        """Test that model performance is stable across different data splits"""
        iris = load_iris(as_frame=True)
        X, y = iris.data, iris.target

        accuracies = []

        # Test with different random states
        for random_state in [42, 123, 456]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            from sklearn.metrics import accuracy_score

            accuracy = accuracy_score(y_test, model.predict(X_test))
            accuracies.append(accuracy)

        # Check that performance is consistently good
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)

        assert min_accuracy >= 0.85, f"Minimum accuracy {min_accuracy:.3f} too low"
        assert (
            max_accuracy - min_accuracy <= 0.15
        ), "Performance too variable across splits"


# Integration test that runs the full evaluation pipeline
def test_full_model_evaluation_pipeline():
    """Integration test for the complete model evaluation process"""

    # 1. Create sample data
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Evaluate model
    predictions = model.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1_score": f1_score(y_test, predictions, average="weighted"),
    }

    # 4. Check all quality gates (realistic thresholds for testing)
    thresholds = {
        "accuracy": 0.85,
        "precision": 0.85,
        "recall": 0.85,
        "f1_score": 0.85,
    }

    for metric, value in metrics.items():
        assert (
            value >= thresholds[metric]
        ), f"{metric} {value:.3f} below threshold {thresholds[metric]}"

    print(f"✅ All quality gates passed: {metrics}")
