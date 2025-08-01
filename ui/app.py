# src/ui/app.py
import streamlit as st, requests, pandas as pd
import os

# Get API URL from environment variable or use default
API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.title("🌸 Iris Species Classifier")
sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal width (cm)", 2.0, 5.0, 3.5)
pl = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal width (cm)", 0.1, 3.0, 0.2)

species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

if st.button("Predict"):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"data": [[sl, sw, pl, pw]]},
            timeout=3,
        )
        resp.raise_for_status()
        pred_num = resp.json()["predictions"][0]
        pred_name = species_map.get(pred_num, "Unknown")
        st.success(f"Model predicts species: **{pred_num} ({pred_name})**")
        st.balloons()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Add API status check
if st.sidebar.checkbox("Show API Status"):
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            st.sidebar.success("✅ API is healthy")
        else:
            st.sidebar.error("❌ API health check failed")
    except:
        st.sidebar.error("❌ Cannot connect to API")
