from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from api.schema import IrisFeatures

app = FastAPI(
    title="Iris Classifier API",
    description="Predict iris species with a Production-stage MLflow model",
    version="1.0.0",
)

model = mlflow.pyfunc.load_model(model_uri="models:/iris_classifier@production")

@app.post("/predict")
def predict(features: IrisFeatures):
    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    df = pd.DataFrame(features.data, columns=columns)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
