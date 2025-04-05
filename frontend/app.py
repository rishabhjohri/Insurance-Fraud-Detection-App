import streamlit as st
import requests

st.title("🛡️ Insurance Claim Fraud Detector")

st.write("### Enter Claim Information")
claim_desc = st.text_area("Claim Description", height=100)
age = st.number_input("Customer Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
occupation = st.text_input("Occupation")
coverage = st.number_input("Coverage Amount")
premium = st.number_input("Premium Amount")
claim_history = st.number_input("Claim History", min_value=0, max_value=100)

if st.button("Submit for Analysis"):
    payload = {
        "Claim_Description": claim_desc,
        "Customer_Age": age,
        "Gender": gender,
        "Marital_Status": marital_status,
        "Occupation": occupation,
        "Coverage_Amount": coverage,
        "Premium_Amount": premium,
        "Claim_History": claim_history
    }
    response = requests.post("http://localhost:8000/predict", json=payload)
    if response.ok:
        result = response.json()
        st.success(f"Fraud Prediction: {'FRAUD' if result['fraud'] else 'Not Fraud'}")
        st.write("### Summary:")
        st.info(result["summary"])
    else:
        st.error("Failed to get response from backend")
