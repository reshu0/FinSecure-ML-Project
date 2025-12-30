import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
from typing import Tuple, Dict, Any

class LoanDefaultPredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        # NEW: Added loan_to_income to feature list
        self.feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 'debt_to_income', 'loan_to_income']
        self.target_column = 'is_default'
        
    def load_data(self, db_path: str = None) -> pd.DataFrame:
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_database.db')
        
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM loan_data", conn)
        conn.close()
        
        # NEW: Feature Engineering
        df['loan_to_income'] = df['loan_amount'] / (df['income'] + 1)
        return df
    
    def create_stacking_classifier(self) -> StackingClassifier:
        base_estimators = [
            ('xgboost', XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )),
            ('random_forest', RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # NEW: Pipeline including SMOTE for class balancing
        self.model = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('stacking', self.create_stacking_classifier())
        ])
        
        print("Training Improved ML pipeline...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        return {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred)
        }
    
    def evaluate_model(self, results: Dict[str, Any]):
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    def save_model(self, model_name: str = 'loan_default_model.pkl'):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(models_dir, model_name))
        print(f"Model saved to: {os.path.join(models_dir, model_name)}")

if __name__ == "__main__":
    predictor = LoanDefaultPredictor()
    data = predictor.load_data()
    results = predictor.train_model(data)
    predictor.evaluate_model(results)
    predictor.save_model()