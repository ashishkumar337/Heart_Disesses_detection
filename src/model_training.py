import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os

from src.data_preprocessing import load_and_preprocess_data

def train_base_models(X_train, y_train):
    print("--- Training Base Models with 5-Fold Cross Validation ---")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fitted_models = {}
    cv_scores = {}
    
    for name, model in models.items():
        # Stratified 5-fold CV
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_scores[name] = np.mean(scores)
        print(f"[{name}] CV Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Fit on whole train set
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
    return fitted_models, cv_scores

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    models, scores = train_base_models(X_train, y_train)
    # Saving base models for testing if needed
    os.makedirs("models/base_models", exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"models/base_models/{name.replace(' ', '_')}.pkl")
    print("Base models trained and saved successfully.")
