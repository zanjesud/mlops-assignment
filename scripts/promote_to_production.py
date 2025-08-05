#!/usr/bin/env python3
"""
Promote staging model to production on master branch merge
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import click
import mlflow
from mlflow.tracking import MlflowClient


def display_production_matrix(run_id, model_name, client):
    """Display production performance matrix before promotion"""
    print("\n=== PRODUCTION PROMOTION MATRIX ===")
    print("Evaluating model for production deployment:")
    
    try:
        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        # Display production thresholds
        print("\nProduction Requirements (95% thresholds):")
        print("  - Accuracy: >=95%")
        print("  - Precision: >=95%")
        print("  - Recall: >=95%")
        print("  - F1-Score: >=95%")
        
        # Check current model performance against production thresholds
        production_thresholds = {
            'accuracy': 0.95,
            'precision': 0.95,
            'recall': 0.95,
            'f1_score': 0.95
        }
        
        print("\nCurrent Model Performance:")
        print("\nProduction Readiness Check:")
        all_pass = True
        for metric, threshold in production_thresholds.items():
            value = metrics.get(metric, 0)
            status = 'PASS' if value >= threshold else 'FAIL'
            print(f"  {metric}: {value:.4f} (req: {threshold:.2f}) [{status}]")
            if value < threshold:
                all_pass = False
        
        print(f"\nOverall Production Readiness: {'READY' if all_pass else 'NOT READY'}")
        print("\n" + "="*50)
        
        return all_pass
        
    except Exception as e:
        print(f"Error displaying production matrix: {e}")
        return False


@click.command()
@click.option("--model_name", default="iris_classifier", help="Model name")
@click.option("--force", is_flag=True, help="Force promotion without validation")
def promote_to_production(model_name, force):
    """Promote staging model to production and update workspace model"""

    client = MlflowClient()

    try:
        # Get staging model info
        staging_info_path = "artifacts/staging_model_info.json"
        if not os.path.exists(staging_info_path):
            print("No staging model info found")
            return False

        with open(staging_info_path, "r") as f:
            staging_info = json.load(f)

        model_version = staging_info["version"]

        # Get run ID for matrix display
        run_id = staging_info["run_id"]
        
        # Display production performance matrix
        production_ready = display_production_matrix(run_id, model_name, client)
        
        # Additional production validation (stricter thresholds)
        if not force and not production_ready:
            print("Model does not meet production criteria")
            return False
        elif force:
            print("Force promotion enabled - skipping validation checks")

        # Use model aliases instead of deprecated stages
        try:
            # Archive current production model if exists
            try:
                current_prod_alias = client.get_model_version_by_alias(
                    model_name, "production"
                )
                if current_prod_alias:
                    print(
                        f"Removing production alias from version {current_prod_alias.version}"
                    )
                    client.delete_registered_model_alias(model_name, "production")
            except Exception:
                print("No current production model to archive")

            # Set production alias
            client.set_registered_model_alias(
                name=model_name, alias="production", version=model_version
            )
            print(
                f"Model {model_name} version {model_version} promoted to production (alias)"
            )

        except Exception as alias_error:
            # Fallback to deprecated stages
            print(f"Alias method failed, trying deprecated stages: {alias_error}")
            try:
                # Archive current production model if exists
                prod_versions = client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                if prod_versions:
                    current_prod = prod_versions[0]
                    print(
                        f"Archiving current production model version {current_prod.version}"
                    )
                    client.transition_model_version_stage(
                        name=model_name, version=current_prod.version, stage="Archived"
                    )
            except Exception:
                print("No current production model to archive")

            # Promote to production
            try:
                client.transition_model_version_stage(
                    name=model_name, version=model_version, stage="Production"
                )
                print(
                    f"Model {model_name} version {model_version} promoted to Production (deprecated)"
                )
            except Exception as stage_error:
                if "cannot represent an object" in str(stage_error):
                    print("Serialization warning ignored - promotion likely succeeded")
                else:
                    raise stage_error

        # Export production model to production_model directory
        model_uri = f"models:/{model_name}@production"
        print(f"Updating production_model directory: {model_uri}")

        # Update production_model directory
        production_model_dir = Path("models/production_model")
        if production_model_dir.exists():
            shutil.rmtree(production_model_dir)
        production_model_dir.mkdir(parents=True, exist_ok=True)

        # Download and copy model files
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = mlflow.artifacts.download_artifacts(
                model_uri, dst_path=temp_dir
            )

            # Copy all model files
            for item in Path(model_path).iterdir():
                if item.is_file():
                    shutil.copy2(item, production_model_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, production_model_dir / item.name)

        # Create model info file
        model_info = {
            "model_name": model_name,
            "version": model_version,
            "stage": "Production",
            "model_uri": model_uri,
            "promoted_to_production": datetime.now().isoformat(),
            "run_id": staging_info["run_id"],
        }

        with open(production_model_dir / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"Production model updated at: {production_model_dir}")

        # Log promotion event
        promotion_log = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "version": model_version,
            "run_id": staging_info["run_id"],
            "stage": "Production",
            "promoted_by": "cicd_pipeline",
            "forced": force,
        }

        os.makedirs("logs", exist_ok=True)
        with open("logs/promotion_history.json", "a") as f:
            f.write(f"{json.dumps(promotion_log)}\n")

        return True

    except Exception as e:
        print(f"Error promoting to production: {e}")
        return False


if __name__ == "__main__":
    promote_to_production()
