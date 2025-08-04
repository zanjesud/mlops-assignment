#!/usr/bin/env python3
"""
Promote model to staging after successful training
"""
import json
import os
from datetime import datetime

import click
import mlflow
from mlflow.tracking import MlflowClient


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
        
        # Promote to staging
        client.transition_model_version_stage(
            name=model_name,
            version=target_version.version,
            stage="Staging"
        )
        
        print(f"Model {model_name} version {target_version.version} promoted to Staging")
        
        # Log promotion event
        promotion_log = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "version": target_version.version,
            "run_id": run_id,
            "stage": "Staging",
            "promoted_by": "cicd_pipeline"
        }
        
        os.makedirs("logs", exist_ok=True)
        with open("logs/promotion_history.json", "a") as f:
            f.write(f"{json.dumps(promotion_log)}\n")
        
        # Save staging info for production promotion
        staging_info = {
            "model_name": model_name,
            "version": target_version.version,
            "run_id": run_id,
            "promoted_to_staging": datetime.now().isoformat()
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