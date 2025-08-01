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

Instrumentator().instrument(app).expose(app, include_in_schema=False)
# http://127.0.0.1:8000/metrics  enpoints for checking matrix


logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

conn = sqlite3.connect("logs/predictions.db", check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS logs (ts TEXT, features TEXT, preds TEXT);")


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Iris Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "metrics": "/metrics",
    }


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        # model = mlflow.pyfunc.load_model(
        #     model_uri="models:/iris_classifier@production"
        # )
        model = mlflow.pyfunc.load_model(model_uri="models/production_model")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return {"error": "Model not found"}

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

    return {"predictions": preds.tolist()}
