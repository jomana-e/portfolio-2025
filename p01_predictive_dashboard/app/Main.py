import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📉 Customer Churn Predictor")
st.write("Use this dashboard to estimate the likelihood of a telecom customer churning.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("../models/churn_pipeline.joblib")
    with open("../models/feature_columns.json") as f:
        feature_cols = json.load(f)
    with open("../models/feature_schema.json") as f:
        schema = json.load(f)
    return model, feature_cols, schema

model, feature_cols, schema = load_artifacts()
cat_cols = schema["categorical"]
num_cols = schema["numerical"]

st.sidebar.header("🧾 Customer Info")

age = st.sidebar.slider("Age", 18, 98, 35)
tenure = st.sidebar.slider("Tenure in Months", 0, 72, 12)
monthly_charge = st.sidebar.slider("Monthly Charge ($)", 0, 200, 70)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check", "Electronic Check"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"])
tech_support = st.sidebar.selectbox("Premium Tech Support", ["Yes", "No"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])

input_data = {
    "Age": age,
    "Tenure in Months": tenure,
    "Monthly Charge": monthly_charge,
    "Contract": contract,
    "Payment Method": payment_method,
    "Dependents": dependents,
    "Partner": partner,
    "Gender": gender,
    "Internet Service": internet_service,
    "Premium Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Online Security": online_security,
    "Online Backup": online_backup
}

input_df = pd.DataFrame([input_data])

st.subheader("📊 Input Summary")
st.dataframe(input_df, use_container_width=True)

if st.button("🔮 Predict Churn"):
    try:
        full_input = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in input_df.columns:
                full_input.loc[0, col] = input_df.loc[0, col]
            elif col in num_cols:
                full_input.loc[0, col] = 0
            else:
                full_input.loc[0, col] = "Unknown"

        for col in num_cols:
            full_input[col] = pd.to_numeric(full_input[col], errors="coerce").fillna(0)

        pred_prob = model.predict_proba(full_input)[0][1]
        st.success(f"📈 Churn Probability: {pred_prob:.2%}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.caption(f"Model expects: {feature_cols}")
        st.caption(f"App provided: {list(input_df.columns)}")
