import os
import time

import requests
import streamlit as st

from diamonds.data import load_data

@st.cache_data
def load_diamonds_data():
    with st.spinner("Loading data..."):
        time.sleep(2)  # Simulate loading time
        return load_data()

df_diamonds = load_diamonds_data()

API_HOST = os.environ.get("API_HOST", "http://localhost:8888").rstrip("/")


st.markdown("## Predict Price")

with st.form("predict_form"):
    carat = st.slider("Carat", min_value=0.2, max_value=5.0, step=0.1)
    cut = st.selectbox("Cut", options=df_diamonds["cut"].unique())
    color = st.selectbox("Color", options=df_diamonds["color"].unique())
    clarity = st.selectbox("Clarity", options=df_diamonds["clarity"].unique())
    depth = st.slider("Depth", min_value=50.0, max_value=70.0, step=0.1)
    table = st.slider("Table", min_value=50.0, max_value=70.0, step=0.1)
    x = st.number_input("X (length in mm)", min_value=0.0, max_value=10.0, step=0.1)
    y = st.number_input("Y (width in mm)", min_value=0.0, max_value=10.0, step=0.1)
    z = st.number_input("Z (depth in mm)", min_value=0.0, max_value=10.0, step=0.1)
    predict_clicked = st.form_submit_button("Predict")

if predict_clicked:
    st.snow()
    url = f"{API_HOST}/predict_one"
    data = {
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
    }
    try:
        with st.spinner("Calling prediction API..."):
            response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        payload = response.json()
        prediction = payload.get("price")
    except requests.RequestException as exc:
        st.error("Prediction API is unreachable. Please try again.")
        st.write(str(exc))
    except ValueError:
        st.error("Prediction API returned an invalid JSON response.")
    else:
        if prediction is None:
            st.error("Prediction API response is missing 'price'.")
            st.write(payload)
        else:
            st.success(f"Predicted Price: ${prediction:.2f}")

