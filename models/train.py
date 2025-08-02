import json
import os
import time

import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "--model_type",
    type=click.Choice(["logreg", "rf"]),
    default="logreg",
    help="Type of model to train",
)
@click.option("--test_size", type=float, default=0.2, help="Test set size")
@click.option(
    "--random_state", type=int, default=42, help="Random state for reproducibility"
)
def train(model_type, test_size, random_state):
    """
    Train a machine learning model and log experiments with MLflow.
    """
    # Set up MLflow experiment
    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run():
        # Log essential parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "test_size": test_size,
                "random_state": random_state,
            }
        )

        # Load and prepare data
        df = pd.read_csv("data/raw/iris.csv")
        X, y = df.drop("target", axis=1), df["target"]

        # Log dataset info
        mlflow.log_params(
            {
                "dataset_rows": len(df),
                "dataset_columns": len(df.columns),
                "features_count": len(X.columns),
                "target_classes": len(y.unique()),
            }
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Log split info
        mlflow.log_params({"train_samples": len(X_train), "test_samples": len(X_test)})

        # Model selection and training
        if model_type == "logreg":
            model = LogisticRegression(max_iter=200, random_state=random_state)
            mlflow.log_param("max_iter", 200)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            mlflow.log_param("n_estimators", 100)

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics
        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "training_time_seconds": float(training_time),
            }
        )

        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            columns=[f"pred_{i}" for i in range(len(cm))],
            index=[f"true_{i}" for i in range(len(cm))],
        )
        cm_path = "confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True)
        mlflow.log_artifact(cm_path)

        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)

        # Log feature importance for Random Forest
        if model_type == "rf":
            feature_importance_df = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            importance_path = "feature_importance.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            # Log top 3 feature importances
            for i, (feature, importance) in enumerate(
                feature_importance_df.head(3).values
            ):
                sanitized_feature = (
                    feature.replace("(", "")
                    .replace(")", "")
                    .replace(" ", "_")
                    .replace("-", "_")
                )
                mlflow.log_metric(
                    f"top_feature_{i + 1}_{sanitized_feature}", importance
                )

        signature = infer_signature(X_test, y_pred)
        # Log model
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name="iris_classifier" if accuracy > 0.92 else None,
        )

        # Save test data for later use
        os.makedirs("data/processed", exist_ok=True)
        test_data_path = "data/processed/test_data.pkl"
        joblib.dump((X_test, y_test), test_data_path)
        mlflow.log_artifact(test_data_path)

        if accuracy > 0.92:
            print("Model registered successfully!")


if __name__ == "__main__":
    train()
