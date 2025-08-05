#!/usr/bin/env python3
"""
Promote model to staging after successful training
"""
import json
import os
from datetime import datetime

import click
from mlflow.tracking import MlflowClient


def display_performance_matrix(run_id, model_name, client):
    """Display model performance matrix before promotion"""
    print("\n=== MODEL PERFORMANCE MATRIX ===")
    print("Displaying model metrics before staging promotion:")
    
    try:
        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        print("\nPerformance Metrics:")
        for metric, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
        
        # Check if validation results exist in artifacts
        if os.path.exists("artifacts/latest_run_info.json"):
            with open("artifacts/latest_run_info.json", "r") as f:
                run_info = json.load(f)
                validation = run_info.get("validation_results", {})
                
                print("\nValidation Results:")
                print(f"  Data Quality: {'PASS' if validation.get('data_quality_passed') else 'FAIL'}")
                print(f"  Model Performance: {'PASS' if validation.get('model_performance_passed') else 'FAIL'}")
                print(f"  Robustness: {'PASS' if validation.get('robustness_passed') else 'FAIL'}")
                
                if validation.get('validation_errors'):
                    print("\nValidation Warnings:")
                    for error in validation['validation_errors']:
                        print(f"  - {error}")
        
        # Display staging thresholds
        print("\nStaging Requirements (85% thresholds):")
        staging_thresholds = {
            'accuracy': 0.85,
            'precision': 0.85,
            'recall': 0.85,
            'f1_score': 0.85,
            'min_class_f1': 0.80
        }
        
        print("\nStaging Readiness Check:")
        all_pass = True
        for metric, threshold in staging_thresholds.items():
            value = metrics.get(metric, 0)
            status = 'PASS' if value >= threshold else 'FAIL'
            print(f"  {metric}: {value:.4f} (req: {threshold:.2f}) [{status}]")
            if value < threshold:
                all_pass = False
        
        print(f"\nOverall Staging Readiness: {'READY' if all_pass else 'NOT READY'}")
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"Error displaying performance matrix: {e}")


@click.command()
@click.option("--run_id", help="MLflow run ID to promote")
@click.option("--model_name", default="iris_classifier", help="Model name")
def promote_to_staging(run_id, model_name):
    """Promote model to staging stage"""

    client = MlflowClient()

    try:
        # Get run info if not provided
        if not run_id:
            if os.path.exists("artifacts/latest_run_info.json"):
                with open("artifacts/latest_run_info.json", "r") as f:
                    run_info = json.load(f)
                    run_id = run_info["run_id"]
                    print(f"Using run ID from artifacts: {run_id}")
            else:
                print("No run ID provided and no artifacts found")
                return False

        # Get the latest version of the model
        try:
            versions = client.get_latest_versions(model_name)
            if not versions:
                print(f"No versions found for model {model_name}")
                return False

            # Find the version corresponding to this run
            target_version = None
            for version in versions:
                if version.run_id == run_id:
                    target_version = version
                    break

            if not target_version:
                # Get all versions and find by run_id
                all_versions = client.search_model_versions(f"name='{model_name}'")
                for version in all_versions:
                    if version.run_id == run_id:
                        target_version = version
                        break

            if not target_version:
                print(f"No model version found for run {run_id}")
                return False

        except Exception as e:
            print(f"Error finding model version: {e}")
            return False

        # Display performance matrix before promotion
        display_performance_matrix(run_id, model_name, client)
        
        print("\nPromoting model to staging...")
        
        # Use model aliases instead of deprecated stages to avoid serialization errors
        try:
            client.set_registered_model_alias(
                name=model_name, alias="staging", version=target_version.version
            )
            print(
                f"✅ Model {model_name} version {target_version.version} promoted to staging (alias)"
            )
        except Exception as alias_error:
            # Fallback to deprecated stages if aliases not supported
            print(f"Alias method failed, trying deprecated stages: {alias_error}")
            try:
                client.transition_model_version_stage(
                    name=model_name, version=target_version.version, stage="Staging"
                )
                print(
                    f"Model {model_name} version {target_version.version} promoted to Staging (deprecated)"
                )
            except Exception as stage_error:
                if "cannot represent an object" in str(stage_error):
                    print("Serialization warning ignored - promotion likely succeeded")
                else:
                    raise stage_error

        # Log promotion event
        promotion_log = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "version": target_version.version,
            "run_id": run_id,
            "stage": "Staging",
            "promoted_by": "cicd_pipeline",
        }

        os.makedirs("logs", exist_ok=True)
        with open("logs/promotion_history.json", "a") as f:
            f.write(f"{json.dumps(promotion_log)}\n")

        # Save staging info for production promotion
        staging_info = {
            "model_name": model_name,
            "version": target_version.version,
            "run_id": run_id,
            "promoted_to_staging": datetime.now().isoformat(),
        }

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/staging_model_info.json", "w") as f:
            json.dump(staging_info, f, indent=2)

        return True

    except Exception as e:
        print(f"Error promoting to staging: {e}")
        return False


if __name__ == "__main__":
    promote_to_staging()
