import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess_data(data_path="data/heart-disease.csv"):
    print("--- Running Data Preprocessing ---")
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Identify column types
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    # Filter to only include columns that actually exist in the dataframe
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data BEFORE applying SMOTE to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Original Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Fit the preprocessor on training data, transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    print("Saved preprocessor to 'models/preprocessor.pkl'")
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"Train shape after SMOTE: {X_train_resampled.shape}")
    
    return X_train_resampled, X_test_processed, y_train_resampled, y_test, X.columns

if __name__ == "__main__":
    load_and_preprocess_data()
