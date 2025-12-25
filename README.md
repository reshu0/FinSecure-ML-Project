# 🛡️ FinSecure: End-to-End Loan Default Risk Engine

An automated Machine Learning pipeline built for the FinTech sector to predict credit default risk using advanced ensemble techniques.

## 🚀 Key Features
* **SQL Data Pipeline:** Automated ingestion from a local SQLite database.
* **Advanced ML Ensemble:** Implements a **Stacking Classifier** combining XGBoost, Random Forest, and Logistic Regression.
* **Imbalance Handling:** Optimized for imbalanced financial data using Scikit-Learn preprocessing.
* **Live Dashboard:** Interactive Streamlit UI for real-time risk assessment and feature importance visualization.

## 📊 Model Performance
* **AUC-ROC:** ~0.6163
* **F1-Score:** 0.66 (Non-Default), 0.46 (Default)

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **ML Libraries:** Scikit-Learn, XGBoost, Joblib
* **Data/DB:** Pandas, SQLite, SQLAlchemy
* **UI/UX:** Streamlit

## ⚙️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/model.py`
3. Launch dashboard: `streamlit run app.py`