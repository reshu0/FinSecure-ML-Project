import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_database():
    """Create SQLite database and populate with synthetic loan default data"""
    
    # Database path
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'loan_database.db')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS loan_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER NOT NULL,
            income REAL NOT NULL,
            credit_score INTEGER NOT NULL,
            loan_amount REAL NOT NULL,
            debt_to_income REAL NOT NULL,
            is_default INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Generate synthetic data
    np.random.seed(42)
    n_records = 10000
    
    # Generate realistic loan data
    data = {
        'age': np.random.randint(18, 75, n_records),
        'income': np.random.lognormal(10.5, 0.5, n_records),  # Log-normal distribution for income
        'credit_score': np.random.randint(300, 850, n_records),
        'loan_amount': np.random.lognormal(9.5, 0.8, n_records),  # Log-normal for loan amounts
        'debt_to_income': np.random.beta(2, 5, n_records) * 0.6,  # DTI ratio between 0-60%
    }
    
    df = pd.DataFrame(data)
    
    # Calculate default probability based on risk factors
    # Higher default probability for: lower credit score, higher DTI, lower income, higher loan amount
    credit_score_factor = (850 - df['credit_score']) / 550
    dti_factor = df['debt_to_income'] / 0.6
    income_factor = np.exp(-df['income'] / 100000)
    loan_factor = df['loan_amount'] / 500000
    
    default_probability = (credit_score_factor * 0.4 + 
                         dti_factor * 0.3 + 
                         income_factor * 0.2 + 
                         loan_factor * 0.1)
    
    # Add some randomness and cap between 0.05 and 0.95
    default_probability = np.clip(default_probability + np.random.normal(0, 0.1, n_records), 0.05, 0.95)
    
    # Generate default labels
    df['is_default'] = (np.random.random(n_records) < default_probability).astype(int)
    
    # Clean up data - ensure reasonable ranges
    df['income'] = np.clip(df['income'], 20000, 500000)
    df['loan_amount'] = np.clip(df['loan_amount'], 5000, 1000000)
    df['debt_to_income'] = np.clip(df['debt_to_income'], 0.01, 0.6)
    
    # Insert data into database
    df.to_sql('loan_data', conn, if_exists='replace', index=False)
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at: {db_path}")
    print(f"Generated {n_records} synthetic loan records")
    print(f"Default rate: {df['is_default'].mean():.2%}")
    print("\nSample data:")
    print(df.head())
    print("\nData statistics:")
    print(df.describe())

if __name__ == "__main__":
    create_database()
