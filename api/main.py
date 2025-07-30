from fastapi import FastAPI
from pydantic import BaseModel, conlist, field_validator,Field
import mlflow.pyfunc

from typing import List

class IrisFeatures(BaseModel):
    data: List[List[float]] = Field(..., min_items=1)

    @field_validator('data')
    def check_inner_list_length(cls, v):
        if not all(len(inner) == 4 for inner in v):
            raise ValueError('Each inner list must have exactly 4 floats')
        return v

app = FastAPI(
    title="Iris Classifier API",
    description="Predict iris species with a Production-stage MLflow model",
    version="1.0.0",
)

model = mlflow.pyfunc.load_model(model_uri="models:/iris_classifier@production")

import pandas as pd  # Add this import at the top

@app.post("/predict")
def predict(features: IrisFeatures):
    columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    df = pd.DataFrame(features.data, columns=columns)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
