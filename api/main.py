from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from api.schema import IrisFeatures
import logging, sqlite3, json, datetime as dt

app = FastAPI(
    title="Iris Classifier API",
    description="Predict iris species with a Production-stage MLflow model",
    version="1.0.0",
)

model = mlflow.pyfunc.load_model(model_uri="models:/iris_classifier@production")

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

conn = sqlite3.connect("logs/predictions.db", check_same_thread=False)
conn.execute(
    "CREATE TABLE IF NOT EXISTS logs (ts TEXT, features TEXT, preds TEXT);"
)

@app.post("/predict")
def predict(features: IrisFeatures):
    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    df = pd.DataFrame(features.data, columns=columns)
    preds = model.predict(df)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO logs VALUES (?,?,?)",
        (now, json.dumps(features.data), json.dumps(preds.tolist())),
    )
    conn.commit()
    logging.info(f"Prediction logged for {json.dumps(features.data)} as {json.dumps(preds.tolist())}")
    
    return {"predictions": preds.tolist()}
