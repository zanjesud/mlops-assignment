# src/ui/app.py
import streamlit as st, requests, pandas as pd
st.title("🌸 Iris Species Classifier")
sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal width (cm)", 2.0, 5.0, 3.5)
pl = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal width (cm)", 0.1, 3.0, 0.2)

species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

if st.button("Predict"):
    resp = requests.post(
        "http://localhost:8000/predict",
        json={"data": [[sl, sw, pl, pw]]},
        timeout=3,
    )
    pred_num = resp.json()["predictions"][0]
    pred_name = species_map.get(pred_num, "Unknown")
    st.success(f"Model predicts species: **{pred_num} ({pred_name})**")
    st.balloons()
