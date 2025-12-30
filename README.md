# 🛡️ FinSecure: Advanced Loan Default Risk Engine

An end-to-end Machine Learning pipeline built for the FinTech sector to predict credit default risk using high-signal synthetic data and advanced ensemble stacking.

## 🚀 Key Features
* **High-Signal Data Pipeline:** Automated generation and ingestion of 15,000+ synthetic financial records using **SQLite** and **SQLAlchemy**.
* **Advanced ML Ensemble:** Implements a **Triple-Stacking Classifier** combining **XGBoost**, **Random Forest**, and **Gradient Boosting**.
* **Feature Engineering:** Includes **Polynomial Feature interactions** and custom ratios like **Loan-to-Income** to capture complex risk patterns.
* **Imbalance Handling:** Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure robust performance on imbalanced default data.
* **Live Dashboard:** Interactive **Streamlit** UI for real-time risk assessment and probability scoring.

## 📊 Model Performance
* **AUC-ROC Score:** **0.6732** .
* **Non-Default F1-Score:** ~0.63.
* **Default F1-Score:** ~0.55.
* **Accuracy:** ~60%.

## 🛠️ Tech Stack
* **Language:** Python 3.10+.
* **ML Libraries:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE).
* **Data/DB:** Pandas, NumPy, SQLite, SQLAlchemy.
* **Deployment:** Streamlit, Joblib.

## ⚙️ How to Run
1. **Install dependencies:** ```bash
   pip install -r requirements.txt
   pip install imbalanced-learn
