#!/usr/bin/env python3
"""
Model evaluation script for CI/CD pipeline
"""

import json
import os

import click
import joblib
import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


@click.command()
@click.option(
    "--model_name", default="iris_classifier", help="Name of the model to evaluate"
)
@click.option("--stage", default="production", help="Model stage to evaluate")
def evaluate_model(model_name, stage):
    """
    Evaluate a trained model and generate performance metrics.
    """
    try:
        # Load test data
        test_data_path = "data/processed/test_data.pkl"
        if not os.path.exists(test_data_path):
            print(f"❌ Test data not found at {test_data_path}")
            print("Please run training first to generate test data.")
            return

        X_test, y_test = joblib.load(test_data_path)
        print(f"✅ Loaded test data: {X_test.shape[0]} samples")

        # Load model
        model_uri = f"models:/{model_name}@{stage}"
        print(f"🔍 Loading model from: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        # Save metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "model_name": model_name,
            "stage": stage,
            "test_samples": len(X_test),
        }

        # Save detailed report
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save performance summary
        with open("performance_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Print results
        print("\n📊 Model Performance Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Check performance thresholds
        thresholds = {
            "accuracy": 0.92,
            "f1_score": 0.90,
            "precision": 0.90,
            "recall": 0.90,
        }

        print("\n🎯 Performance Thresholds:")
        all_passed = True
        for metric, threshold in thresholds.items():
            value = metrics[metric]
            passed = value >= threshold
            status = "✅" if passed else "❌"
            print(
                f"{status} {metric.capitalize()}: {value:.4f} (threshold: {threshold:.2f})"
            )
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 All performance thresholds met!")
        else:
            print("\n⚠️  Some performance thresholds not met!")

        return metrics

    except Exception as e:
        print(f"❌ Error during model evaluation: {e}")
        return None


if __name__ == "__main__":
    evaluate_model()
