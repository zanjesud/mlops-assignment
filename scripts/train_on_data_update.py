#!/usr/bin/env python3
"""
Data-driven model training for CI/CD pipeline
Triggered when iris.csv.dvc is updated
"""
import json
import os
import time
from datetime import datetime

import click
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


@click.command()
@click.option("--data_path", default="data/raw/iris.csv", help="Path to training data")
@click.option(
    "--experiment_name", default="iris-classifier-cicd", help="MLflow experiment name"
)
@click.option(
    "--model_name", default="iris_classifier", help="Model name for registration"
)
def train_model(data_path, experiment_name, model_name):
    """Train model on updated data and register if performance is good"""

    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        print(f"Starting training run: {run.info.run_id}")

        # Log data source info
        data_info = {
            "data_path": data_path,
            "data_updated": datetime.now().isoformat(),
            "triggered_by": "data_update",
        }
        mlflow.log_params(data_info)

        # Load and validate data
        try:
            df = pd.read_csv(data_path)
            print(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

        # Data validation
        required_columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
            "target",
        ]
        if not all(col in df.columns for col in required_columns):
            print(f"Missing required columns. Expected: {required_columns}")
            return None

        # Prepare data
        X = df.drop("target", axis=1)
        y = df["target"]

        # Log dataset statistics
        mlflow.log_params(
            {
                "dataset_rows": len(df),
                "dataset_features": len(X.columns),
                "target_classes": len(y.unique()),
                "class_distribution": dict(y.value_counts()),
            }
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model with optimized parameters
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "training_time": training_time,
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model with signature
        signature = infer_signature(X_test, y_pred)

        # Register model if it meets staging criteria
        staging_thresholds = {
            "accuracy": 0.90,
            "precision": 0.90,
            "recall": 0.90,
            "f1_score": 0.90,
        }

        meets_criteria = all(
            metrics[metric] >= threshold
            for metric, threshold in staging_thresholds.items()
        )

        if meets_criteria:
            # Register model
            mlflow.sklearn.log_model(
                model, "model", signature=signature, registered_model_name=model_name
            )

            # Save run info for promotion
            run_info = {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "meets_staging_criteria": True,
                "timestamp": datetime.now().isoformat(),
            }

            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/latest_run_info.json", "w") as f:
                json.dump(run_info, f, indent=2)

            print(f"Model registered! Run ID: {run.info.run_id}")
            print(f"Metrics: {metrics}")
            return run.info.run_id
        else:
            print("Model does not meet staging criteria")
            print(f"Current metrics: {metrics}")
            print(f"Required thresholds: {staging_thresholds}")
            return None


if __name__ == "__main__":
    train_model()
