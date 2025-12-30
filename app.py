import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="FinSecure Risk Dashboard", layout="wide")

st.title("🛡️ FinSecure: Loan Risk Assessment Engine")
st.markdown("Automated Credit Risk Prediction using SMOTE + Stacking Ensemble")

MODEL_PATH = "models/loan_default_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model not found. Please run 'python src/model.py' first.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("User Financial Profile")
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=15000)
    debt_to_income = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.2)

    if st.button("Predict Default Risk"):
        # NEW: Calculate the engineered feature
        loan_to_income = loan_amount / (income + 1)
        
        # Prepare data with all 6 features
        input_data = pd.DataFrame(
            [[age, income, credit_score, loan_amount, debt_to_income, loan_to_income]], 
            columns=['age', 'income', 'credit_score', 'loan_amount', 'debt_to_income', 'loan_to_income']
        )
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"High Risk: Default Likely ({probability:.2%})")
        else:
            st.success(f"Low Risk: Approval Recommended ({probability:.2%})")

with col2:
    st.subheader("Model Insights")
    st.info("Upgraded Model: Uses SMOTE class balancing and Feature Engineering.")
    st.write("Target Performance: AUC-ROC Improvement through Ensemble Stacking.")