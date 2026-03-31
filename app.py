import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("fraud_pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ===============================
# UI
# ===============================
st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's Fraud or Genuine.")

st.markdown("---")

# ===============================
# INPUT FIELDS (ONLY 3)
# ===============================
st.subheader("📥 Transaction Input")

time = st.number_input("⏱️ Time", value=0.0)
amount = st.number_input("💰 Amount", value=0.0)
transaction = st.number_input("🔢 Transaction ID", value=0)

# ===============================
# CREATE FULL INPUT FOR MODEL
# ===============================
input_data = {"Time": time}

# Fill V1–V28 with 0
for i in range(1, 29):
    input_data[f"V{i}"] = 0.0

input_data["Amount"] = amount

input_df = pd.DataFrame([input_data])

# ===============================
# PREDICTION
# ===============================
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)

    st.markdown("---")

    if prediction[0] == 0:
        st.success("✅ Transaction is Genuine")
    else:
        st.error("⚠️ Transaction is Fraudulent")
