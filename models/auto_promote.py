#!/usr/bin/env python3
"""
Automated Model Promotion Script
Can be used in CI/CD pipelines for automatic model promotion based on performance.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AutoModelPromoter:
    """Automated model promotion with performance validation"""

    def __init__(self, model_name="iris_classifier"):
        self.model_name = model_name
        self.client = MlflowClient()

        # Performance thresholds for automatic promotion
        self.thresholds = {
            "staging": {
                "accuracy": 0.85,
                "precision": 0.85,
                "recall": 0.85,
                "f1_score": 0.85,
            },
            "production": {
                "accuracy": 0.95,
                "precision": 0.95,
                "recall": 0.95,
                "f1_score": 0.95,
            },
        }

    def evaluate_model(self, model_uri, test_data_path="data/processed/test_data.pkl"):
        """Evaluate model performance"""
        try:
            # Load test data
            X_test, y_test = joblib.load(test_data_path)

            # Load model
            model = mlflow.sklearn.load_model(model_uri)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1_score": f1_score(y_test, y_pred, average="weighted"),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

    def check_performance_thresholds(self, metrics, stage):
        """Check if metrics meet promotion thresholds"""
        thresholds = self.thresholds[stage]

        for metric, threshold in thresholds.items():
            if metrics[metric] < threshold:
                logger.warning(
                    f"{metric} ({metrics[metric]:.4f}) below threshold ({threshold})"
                )
                return False

        return True

    def auto_promote_run(self, run_id, target_stage="staging"):
        """Automatically promote a model from a run based on performance"""
        try:
            model_uri = f"runs:/{run_id}/model"

            # Evaluate model performance
            metrics = self.evaluate_model(model_uri)
            if metrics is None:
                return False, "Model evaluation failed"

            logger.info(f"Model Performance: {metrics}")

            # Check if performance meets thresholds
            if not self.check_performance_thresholds(metrics, target_stage):
                return False, f"Performance below {target_stage} thresholds"

            # Register model if not already registered
            try:
                self.client.get_registered_model(self.model_name)
            except Exception:
                logger.info(f"Registering new model: {self.model_name}")
                mlflow.register_model(model_uri, self.model_name)

            # Get the latest version
            latest_version = self.client.get_latest_versions(self.model_name)[0]

            # Promote to target stage
            stage_name = target_stage.capitalize()
            self.client.transition_model_version_stage(
                name=self.model_name, version=latest_version.version, stage=stage_name
            )

            logger.info(
                f"✅ Model automatically promoted to {stage_name} - Version {latest_version.version}"
            )

            # Log promotion event
            promotion_log = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "model_name": self.model_name,
                "version": latest_version.version,
                "stage": stage_name,
                "metrics": metrics,
                "automated": True,
            }

            os.makedirs("logs", exist_ok=True)
            with open("logs/auto_promotion_history.json", "a") as f:
                f.write(f"{json.dumps(promotion_log)}\n")

            return True, f"Successfully promoted to {stage_name}"

        except Exception as e:
            logger.error(f"Error in auto promotion: {e}")
            return False, str(e)

    def auto_promote_staging_to_production(self, force=False):
        """Automatically promote staging model to production if it meets criteria"""
        try:
            # Get staging model
            staging_versions = self.client.get_latest_versions(
                self.model_name, stages=["Staging"]
            )
            if not staging_versions:
                return False, "No model found in Staging stage"

            staging_version = staging_versions[0]
            model_uri = f"models:/{self.model_name}/{staging_version.version}"

            # Evaluate model performance
            metrics = self.evaluate_model(model_uri)
            if metrics is None:
                return False, "Model evaluation failed"

            logger.info(f"Staging Model Performance: {metrics}")

            # Check if performance meets production thresholds
            if (
                not self.check_performance_thresholds(metrics, "production")
                and not force
            ):
                return False, "Performance below production thresholds"

            # Archive current production model if exists
            prod_versions = self.client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            if prod_versions:
                current_prod_version = prod_versions[0].version
                logger.info(
                    f"Archiving current production model (Version {current_prod_version})"
                )
                self.client.transition_model_version_stage(
                    name=self.model_name, version=current_prod_version, stage="Archived"
                )

            # Promote to Production
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=staging_version.version,
                stage="Production",
            )

            logger.info(
                f"✅ Model automatically promoted to Production - Version {staging_version.version}"
            )

            # Log promotion event
            promotion_log = {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "version": staging_version.version,
                "previous_production_version": (
                    current_prod_version if prod_versions else None
                ),
                "metrics": metrics,
                "automated": True,
                "forced": force,
            }

            os.makedirs("logs", exist_ok=True)
            with open("logs/auto_promotion_history.json", "a") as f:
                f.write(f"{json.dumps(promotion_log)}\n")

            return True, "Successfully promoted to Production"

        except Exception as e:
            logger.error(f"Error in auto promotion to production: {e}")
            return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Automated Model Promotion")
    parser.add_argument(
        "--action",
        choices=["promote_run", "promote_staging"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument("--run-id", help="MLflow run ID (required for promote_run)")
    parser.add_argument("--model-name", default="iris_classifier", help="Model name")
    parser.add_argument(
        "--target-stage",
        choices=["staging", "production"],
        default="staging",
        help="Target stage for promotion",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force promotion even if validation fails"
    )

    args = parser.parse_args()

    promoter = AutoModelPromoter(args.model_name)

    if args.action == "promote_run":
        if not args.run_id:
            logger.error("--run-id is required for promote_run action")
            sys.exit(1)

        success, message = promoter.auto_promote_run(args.run_id, args.target_stage)
        if success:
            logger.info(f"✅ {message}")
        else:
            logger.error(f"❌ {message}")
            sys.exit(1)

    elif args.action == "promote_staging":
        success, message = promoter.auto_promote_staging_to_production(args.force)
        if success:
            logger.info(f"✅ {message}")
        else:
            logger.error(f"❌ {message}")
            sys.exit(1)


if __name__ == "__main__":
    main()
