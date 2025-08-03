import datetime as dt
import json
import logging
import sqlite3

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from api.schema import IrisFeatures

app = FastAPI(
    title="Iris Classifier API",
    description="Predict iris species with a Production-stage MLflow model",
    version="1.0.0",
)

# Custom metrics counters
prediction_counter = 0
species_predictions = {"setosa": 0, "versicolor": 0, "virginica": 0}
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

Instrumentator().instrument(app).expose(app, include_in_schema=False)
# http://127.0.0.1:8000/metrics  enpoints for checking matrix

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

conn = sqlite3.connect("logs/predictions.db", check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS logs (ts TEXT, features TEXT, preds TEXT);")

# Load model once at startup
model = None
try:
    model = mlflow.pyfunc.load_model(model_uri="models/production_model")
    logging.info("Model loaded successfully at startup")
except Exception as e:
    logging.error(f"Error loading model at startup: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_loaded": model is not None,
    }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        logging.error("Model not loaded")
        return {"error": "Model not available"}
    try:
        columns = [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
        df = pd.DataFrame(features.data, columns=columns)
        preds = model.predict(df)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO logs VALUES (?,?,?)",
            (now, json.dumps(features.data), json.dumps(preds.tolist())),
        )
        conn.commit()
        logging.info(
            f"Prediction logged for {json.dumps(features.data)} as {json.dumps(preds.tolist())}"
        )

        # Update custom metrics
        global prediction_counter
        prediction_counter += len(preds)
        for pred in preds:
            species_name = species_map.get(int(pred), "unknown")
            if species_name in species_predictions:
                species_predictions[species_name] += 1

        return {"predictions": preds.tolist()}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}


@app.get("/custom-metrics")
def custom_metrics():
    """Custom metrics endpoint for enhanced monitoring"""
    return {
        "total_predictions": prediction_counter,
        "species_breakdown": species_predictions,
        "model_status": "loaded" if model is not None else "not_loaded",
        "database_status": "connected",
    }
