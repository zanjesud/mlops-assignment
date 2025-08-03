# src/ui/app.py
import os

import requests
import streamlit as st

# Get API URL from environment variable or use default
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🌸 Iris Species Classifier")
sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal width (cm)", 2.0, 5.0, 3.5)
pl = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal width (cm)", 0.1, 3.0, 0.2)

species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

if st.button("Predict"):
    with st.spinner("🤖 Loading model and making prediction..."):
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"data": [[sl, sw, pl, pw]]},
                timeout=10,  # Increased from 3 to 10 seconds
            )
            resp.raise_for_status()
            pred_num = resp.json()["predictions"][0]
            pred_name = species_map.get(pred_num, "Unknown")
            st.success(f"Model predicts species: **{pred_num} ({pred_name})**")
            st.balloons()
        except requests.exceptions.Timeout:
            st.error("⏰ Request timed out. The model is still loading. Please try again in a few seconds.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Please check if the API service is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Add API status check
if st.sidebar.checkbox("Show API Status"):
    try:
        health_resp = requests.get(f"{API_URL}/health", timeout=5)  # Increased from 2 to 5 seconds
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            if health_data.get("model_loaded", False):
                st.sidebar.success("✅ API is healthy (Model loaded)")
            else:
                st.sidebar.warning("⚠️ API is running but model not loaded")
        else:
            st.sidebar.error("❌ API health check failed")
    except Exception:
        st.sidebar.error("❌ Cannot connect to API")
