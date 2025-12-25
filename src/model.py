import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
import joblib
import os
from typing import Tuple, Dict, Any

class LoanDefaultPredictor:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 'debt_to_income']
        self.target_column = 'is_default'
        
    def load_data(self, db_path: str = None) -> pd.DataFrame:
        """Fetch data from SQLite database"""
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_database.db')
        
        try:
            conn = sqlite3.connect(db_path)
            query = f"SELECT {', '.join(self.feature_columns + [self.target_column])} FROM loan_data"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"Data loaded successfully. Shape: {df.shape}")
            print(f"Target distribution: {df[self.target_column].value_counts(normalize=True)}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create preprocessing pipeline with StandardScaler for numeric features"""
        numeric_features = self.feature_columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='passthrough'
        )
        
        return self.preprocessor
    
    def create_stacking_classifier(self) -> StackingClassifier:
        """Create Stacking Classifier with XGBoost and Random Forest as base estimators"""
        # Base estimators
        base_estimators = [
            ('xgboost', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )),
            ('random_forest', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        # Meta-model (final aggregator)
        meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            n_jobs=-1
        )
        
        # Stacking classifier
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1,
            passthrough=True
        )
        
        return stacking_clf
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the complete ML pipeline"""
        print("Training ML pipeline...")
        
        # Split features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Create and fit preprocessing pipeline
        self.create_preprocessing_pipeline()
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Create and train stacking classifier
        self.model = self.create_stacking_classifier()
        
        print("Training stacking classifier...")
        self.model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        # Store results
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        return results
    
    def evaluate_model(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics"""
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
        
        # AUC-ROC Score
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nAUC-ROC Score: {auc_roc:.4f}")
        
        # Additional metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            print("\nFeature Importances:")
            for feature, importance in zip(self.feature_columns, self.model.feature_importances_):
                print(f"{feature}: {importance:.4f}")
        
        metrics = {
            'accuracy': accuracy,
            'auc_roc': auc_roc
        }
        
        return metrics
    
    def save_model(self, model_name: str = 'loan_default_model.pkl') -> str:
        """Save trained model and preprocessor to models folder"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, model_name)
        
        # Save both model and preprocessor
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved successfully to: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str) -> None:
        """Load trained model and preprocessor"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            print(f"Model loaded successfully from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Ensure data has required columns
        if not all(col in data.columns for col in self.feature_columns):
            missing_cols = set(self.feature_columns) - set(data.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Preprocess data
        X_processed = self.preprocessor.transform(data[self.feature_columns])
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        
        return predictions, probabilities

def main():
    """Main function to run the complete ML pipeline"""
    print("Starting Loan Default Prediction ML Pipeline")
    print("="*50)
    
    # Initialize predictor
    predictor = LoanDefaultPredictor()
    
    # Load data
    df = predictor.load_data()
    
    # Train model
    results = predictor.train_model(df)
    
    # Evaluate model
    metrics = predictor.evaluate_model(results)
    
    # Save model
    model_path = predictor.save_model()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Model saved at: {model_path}")

if __name__ == "__main__":
    main()
