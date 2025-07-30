#!/usr/bin/env python3
"""
Model Promotion Script for MLflow
Handles model promotion from Staging to Production with validation checks.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import click
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import sys
from datetime import datetime
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_promotion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelPromoter:
    """Handles model promotion through MLflow stages"""
    
    def __init__(self, model_name="iris_classifier"):
        self.model_name = model_name
        self.client = MlflowClient()
        
    def list_models(self):
        """List all registered models and their versions"""
        try:
            # Use search_registered_models instead of list_registered_models
            models = self.client.search_registered_models()
            logger.info(f"Found {len(models)} registered models")
            
            if not models:
                logger.info("No registered models found. You may need to train and register a model first.")
                return True
            
            for model in models:
                if model.name == self.model_name:
                    try:
                        versions = self.client.search_model_versions(f"name='{model.name}'")
                        # logger.info(versions)
                        logger.info(f"\nModel: {model.name}")
                        for v in self.client.search_model_versions(f"name='{model.name}'"):
                            logger.info(f"Version: {v.version}, Run ID: {v.run_id}")
                            logger.info(f"Aliases: {v.aliases}")
                    except Exception as e:
                        logger.warning(f"Could not get versions for model {model.name}: {e}")
                else:
                    logger.info(f"Model: {model.name} (not the target model)")
            return True
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            logger.info("This might be because:")
            logger.info("1. No models have been registered yet")
            logger.info("2. MLflow tracking server is not running")
            logger.info("3. MLflow is not properly configured")
            return False
    
    def get_model_versions(self, stage=None):
        """Get model versions for a specific stage"""
        try:
            if stage:
                versions = self.client.get_latest_versions(self.model_name, stages=[stage])
            else:
                versions = self.client.get_latest_versions(self.model_name)
            return versions
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def validate_model_performance(self, model_uri, test_data_path="data/processed/test_data.pkl"):
        """Validate model performance against test data"""
        try:
            # Load test data
            X_test, y_test = joblib.load(test_data_path)
            
            # Load model
            model = mlflow.sklearn.load_model(model_uri)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            logger.info(f"Model Performance Validation:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            
            # Define validation criteria
            validation_passed = (
                accuracy >= 0.90 and
                precision >= 0.90 and
                recall >= 0.90 and
                f1 >= 0.90
            )
            
            return validation_passed, {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False, {}
    
    def promote_to_staging(self, run_id):
        """Promote model from None to Staging"""
        try:
            model_uri = f"runs:/{run_id}/model"
            
            # Register model if not already registered
            try:
                self.client.get_registered_model(self.model_name)
            except:
                logger.info(f"Registering new model: {self.model_name}")
                mlflow.register_model(model_uri, self.model_name)
            
            # Get the latest version
            latest_version = self.client.get_latest_versions(self.model_name)[0]
            print(f"latest version : {latest_version.version}")
            print(f"Model URI : {self.model_name}")
            # Transition to Staging
            # print(f"Metrics: {self.client.get_run(run_id).data.metrics}")
            self.client.set_registered_model_alias(
                name=self.model_name,
                version=latest_version.version,
                alias="Staging"
            )
            
            logger.info(f"✅ Model promoted to Staging - Version {latest_version.version}")
            return True
            
        except Exception as e:
            logger.error(f" ❌ Error promoting to staging: {e}")
            return False
            
    def get_model_alias(self, alias):
        try:
            # Returns a list of ModelVersion objects with the given alias
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            return [v for v in versions if alias in getattr(v, "aliases", [])]
        except Exception as e:
            logger.error(f"Error getting model by alias: {e}")
            return []
        
    def promote_to_production(self, version=None, force=False):
        """Promote model from Staging to Production"""
        try:
            # Get staging model
            alias = "Staging"
            staging_versions = self.get_model_alias(alias)
            if not staging_versions:
                logger.error("No model found in Staging stage")           
            
            if version:
                target_version = version
            else:
                target_version = staging_versions[0].version
            
            # Validate model performance
            model_uri = f"models:/{self.model_name}/{target_version}"
            validation_passed, metrics = self.validate_model_performance(model_uri)
            
            if not validation_passed and not force:
                logger.error("❌ Model validation failed. Use --force to override.")
                return False
            
            # Archive current production model if exists
            prod_versions = self.get_model_alias("Production")
            if prod_versions:
                current_prod_version = prod_versions[0].version
                logger.info(f"Archiving current production model (Version {current_prod_version})")
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    version=current_prod_version,
                    alias="Archived"
                )
            
            # Promote to Production
            self.client.set_registered_model_alias(
                name=self.model_name,
                version=target_version,
                alias="Production"
            )
            
            logger.info(f"✅ Model promoted to Production - Version {target_version}")
            
            # Log promotion event
            promotion_log = {
                'timestamp': datetime.now().isoformat(),
                'model_name': self.model_name,
                'version': target_version,
                'previous_production_version': current_prod_version if prod_versions else None,
                'metrics': metrics,
                'validation_passed': validation_passed,
                'forced': force
            }
            
            with open('logs/promotion_history.json', 'a') as f:
                f.write(f"{promotion_log}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting to production: {e}")
            return False
    
    def rollback_production(self, target_version):
        """Rollback production to a specific version"""
        try:
            # Get current production model
            prod_versions = self.get_model_alias("Production")
            if not prod_versions:
                logger.error("No model currently in Production")
                return False
            
            current_prod_version = prod_versions[0].version
            
            # Archive current production
            self.client.set_registered_model_alias(
                name=self.model_name,
                version=current_prod_version,
                alias="Archived"
            )
            
            # Promote target version to production
            self.client.set_registered_model_alias(
                name=self.model_name,
                version=target_version,
                alias="Production"
            )
            
            logger.info(f"✅ Production rolled back to Version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back production: {e}")
            return False
    
    def compare_models(self, version1, version2):
        """Compare performance of two model versions"""
        try:
            # Load test data
            X_test, y_test = joblib.load("data/processed/test_data.pkl")
            
            # Load both models
            model1 = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version1}")
            model2 = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version2}")
            
            # Get predictions
            y_pred1 = model1.predict(X_test)
            y_pred2 = model2.predict(X_test)
            
            # Calculate metrics
            metrics1 = {
                'accuracy': accuracy_score(y_test, y_pred1),
                'precision': precision_score(y_test, y_pred1, average='weighted'),
                'recall': recall_score(y_test, y_pred1, average='weighted'),
                'f1_score': f1_score(y_test, y_pred1, average='weighted')
            }
            
            metrics2 = {
                'accuracy': accuracy_score(y_test, y_pred2),
                'precision': precision_score(y_test, y_pred2, average='weighted'),
                'recall': recall_score(y_test, y_pred2, average='weighted'),
                'f1_score': f1_score(y_test, y_pred2, average='weighted')
            }
            
            logger.info(f"Model Comparison:")
            logger.info(f"Version {version1}: {metrics1}")
            logger.info(f"Version {version2}: {metrics2}")
            
            return metrics1, metrics2
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return None, None

@click.group()
def cli():
    """MLflow Model Promotion CLI"""
    pass

@cli.command()
@click.option("--model-name", default="iris_classifier", help="Name of the model")
def list_models(model_name):
    """List all registered models and their stages"""
    promoter = ModelPromoter(model_name)
    promoter.list_models()

@cli.command()
@click.option("--run-id", required=True, help="MLflow run ID")
@click.option("--model-name", default="iris_classifier", help="Name of the model")
def promote_to_staging(run_id, model_name):
    """Promote model from run to Staging stage"""
    promoter = ModelPromoter(model_name)
    success = promoter.promote_to_staging(run_id)
    if success:
        logger.info("✅ Promotion to Staging successful")
    else:
        logger.error("Promotion to Staging failed")
        sys.exit(1)

@cli.command()
@click.option("--version", type=int, help="Specific version to promote (default: latest staging)")
@click.option("--force", is_flag=True, help="Force promotion even if validation fails")
@click.option("--model-name", default="iris_classifier", help="Name of the model")
def promote_to_production(version, force, model_name):
    """Promote model from Staging to Production"""
    promoter = ModelPromoter(model_name)
    success = promoter.promote_to_production(version, force)
    if success:
        logger.info("✅ Promotion to Production successful")
    else:
        logger.error("❌ Promotion to Production failed")
        sys.exit(1)

@cli.command()
@click.option("--version", type=int, required=True, help="Version to rollback to")
@click.option("--model-name", default="iris_classifier", help="Name of the model")
def rollback(version, model_name):
    """Rollback production to a specific version"""
    promoter = ModelPromoter(model_name)
    success = promoter.rollback_production(version)
    if success:
        logger.info("✅ Rollback successful")
    else:
        logger.error("❌ Rollback failed")
        sys.exit(1)

@cli.command()
@click.option("--version1", type=int, required=True, help="First version to compare")
@click.option("--version2", type=int, required=True, help="Second version to compare")
@click.option("--model-name", default="iris_classifier", help="Name of the model")
def compare(version1, version2, model_name):
    """Compare performance of two model versions"""
    promoter = ModelPromoter(model_name)
    promoter.compare_models(version1, version2)

if __name__ == "__main__":
    cli()