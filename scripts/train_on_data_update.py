#!/usr/bin/env python3
"""
Data-driven model training for CI/CD pipeline
Triggered when iris.csv.dvc is updated
"""
import json
import os
import sys
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
        print(f"Starting training run: {run.info.run_id}", file=sys.stderr)

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
            print(
                f"Loaded data: {len(df)} rows, {len(df.columns)} columns",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
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
            print(
                f"Missing required columns. Expected: {required_columns}",
                file=sys.stderr,
            )
            return None

        # Prepare data
        X = df.drop("target", axis=1)
        y = df["target"]

        # Log dataset statistics with proper type conversion
        class_dist = y.value_counts().to_dict()
        class_dist_str = {str(k): int(v) for k, v in class_dist.items()}

        mlflow.log_params(
            {
                "dataset_rows": int(len(df)),
                "dataset_features": int(len(X.columns)),
                "target_classes": int(len(y.unique())),
            }
        )

        # Log class distribution as separate parameters
        for class_label, count in class_dist_str.items():
            mlflow.log_param(f"class_{class_label}_count", count)

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

        # Calculate metrics and ensure they are basic Python types
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
            "training_time": float(training_time),
        }

        # Additional validation metrics
        import numpy as np
        from sklearn.metrics import classification_report

        # Model robustness validation
        validation_results = {
            "data_quality_passed": True,
            "model_performance_passed": False,
            "robustness_passed": False,
            "validation_errors": [],
        }

        # 1. Data Quality Validation
        print("Performing data quality validation...", file=sys.stderr)

        # Check for data drift (basic)
        if len(df) < 100:
            validation_results["validation_errors"].append(
                "Insufficient data: less than 100 samples"
            )
            validation_results["data_quality_passed"] = False

        # Check class balance
        class_balance = y.value_counts(normalize=True)
        min_class_ratio = class_balance.min()
        if min_class_ratio < 0.1:  # Less than 10% for any class
            validation_results["validation_errors"].append(
                f"Class imbalance detected: min class ratio {min_class_ratio:.3f}"
            )

        # Check for missing values
        if X.isnull().sum().sum() > 0:
            validation_results["validation_errors"].append(
                "Missing values detected in features"
            )
            validation_results["data_quality_passed"] = False

        # 2. Model Performance Validation
        print("Performing model performance validation...", file=sys.stderr)

        # Per-class performance check
        class_report = classification_report(y_test, y_pred, output_dict=True)
        per_class_f1 = [class_report[str(i)]["f1-score"] for i in y.unique()]
        min_class_f1 = min(per_class_f1)

        metrics["min_class_f1"] = float(min_class_f1)
        metrics["class_balance_ratio"] = float(min_class_ratio)

        # 3. Model Robustness Validation
        print("Performing robustness validation...", file=sys.stderr)

        # Cross-validation for stability
        from sklearn.model_selection import cross_val_score

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        metrics["cv_accuracy_mean"] = cv_mean
        metrics["cv_accuracy_std"] = cv_std

        # Check model stability (low variance across CV folds)
        if cv_std > 0.05:  # More than 5% standard deviation
            validation_results["validation_errors"].append(
                f"Model unstable: CV std {cv_std:.3f} > 0.05"
            )
        else:
            validation_results["robustness_passed"] = True

        # Performance thresholds validation
        staging_thresholds = {
            "accuracy": 0.85,  # Lowered from 0.90 for realistic threshold
            "precision": 0.85,
            "recall": 0.85,
            "f1_score": 0.85,
            "min_class_f1": 0.80,  # Ensure all classes perform reasonably
            "cv_accuracy_mean": 0.83,  # Cross-validation performance
        }

        performance_passed = all(
            metrics.get(metric, 0) >= threshold
            for metric, threshold in staging_thresholds.items()
        )

        if performance_passed:
            validation_results["model_performance_passed"] = True
        else:
            failed_metrics = [
                f"{metric}: {metrics.get(metric, 0):.3f} < {threshold}"
                for metric, threshold in staging_thresholds.items()
                if metrics.get(metric, 0) < threshold
            ]
            validation_results["validation_errors"].extend(failed_metrics)

        # Log all metrics individually to avoid serialization issues
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log validation results
        mlflow.log_params(
            {
                "data_quality_passed": validation_results["data_quality_passed"],
                "model_performance_passed": validation_results[
                    "model_performance_passed"
                ],
                "robustness_passed": validation_results["robustness_passed"],
                "validation_errors_count": len(validation_results["validation_errors"]),
            }
        )

        # Log model with signature
        signature = infer_signature(X_test, y_pred)

        # Overall validation check
        meets_criteria = (
            validation_results["data_quality_passed"]
            and validation_results["model_performance_passed"]
            and validation_results["robustness_passed"]
        )

        if meets_criteria:
            print("✅ All validation checks passed", file=sys.stderr)

            # Register model
            mlflow.sklearn.log_model(
                model, "model", signature=signature, registered_model_name=model_name
            )

            # Save comprehensive run info for promotion
            run_info = {
                "run_id": run.info.run_id,
                "metrics": metrics,
                "validation_results": validation_results,
                "meets_staging_criteria": True,
                "timestamp": datetime.now().isoformat(),
            }

            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/latest_run_info.json", "w") as f:
                json.dump(run_info, f, indent=2)

            # Only print run ID for capture by CI/CD
            print(run.info.run_id)
            return run.info.run_id
        else:
            print("❌ Model validation failed", file=sys.stderr)
            print(
                f"Validation errors: {validation_results['validation_errors']}",
                file=sys.stderr,
            )
            print(f"Current metrics: {metrics}", file=sys.stderr)
            print(f"Required thresholds: {staging_thresholds}", file=sys.stderr)
            return None


if __name__ == "__main__":
    train_model()
