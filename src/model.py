import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

class LoanDefaultPredictor:
    def __init__(self):
        self.feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 'debt_to_income', 'loan_to_income']
        self.model = None

    def load_data(self) -> pd.DataFrame:
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_database.db')
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM loan_data", conn)
        conn.close()
        # Feature Engineering: Loan-to-Income is a critical risk indicator
        df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
        return df

    def create_stacking_ensemble(self) -> StackingClassifier:
        base_models = [
            ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
        ]
        # Meta-learner with internal cross-validation for optimal regularization
        return StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegressionCV(cv=5),
            stack_method='predict_proba',
            n_jobs=-1
        )

    def train(self, df: pd.DataFrame):
        X = df[self.feature_columns]
        y = df['is_default']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Performance Pipeline: Interactions -> Scaling -> Balancing -> Ensemble
        self.model = ImbPipeline([
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
            ('stack', self.create_stacking_ensemble())
        ])
        
        print("🚀 Training optimized performance pipeline...")
        self.model.fit(X_train, y_train)
        
        y_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n🎯 FINAL PERFORMANCE")
        print(f"AUC-ROC Score: {auc:.4f}")
        print("-" * 30)
        
        # Save model
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(models_dir, 'loan_default_model.pkl'))

if __name__ == "__main__":
    predictor = LoanDefaultPredictor()
    data = predictor.load_data()
    predictor.train(data)