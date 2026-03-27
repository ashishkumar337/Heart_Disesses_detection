import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_base_models

def evaluate_models(models, X_test, y_test, output_dir="notebooks"):
    print("\n--- Evaluating Models on Test Set ---")
    results = []
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # We need probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC': roc_auc
        })
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (area = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (All Models)')
    plt.legend(loc="lower right")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "roc_auc_all_models.png"))
    plt.close()
    
    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\nBest Model identified based on F1-Score: {best_model_name}")
    return results_df, best_model_name

def tune_best_model(models, best_name, X_train, y_train):
    print(f"\n--- Hyperparameter Tuning: {best_name} ---")
    base_model = models[best_name]
    
    # Generic simple grids matching base model types
    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']},
        'Decision Tree': {'max_depth': [None, 5, 10, 20], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]},
        'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    }
    
    grid = param_grids.get(best_name, {})
    if not grid:
        print("No parameter grid defined for this model. Skipping tuning.")
        return base_model
        
    gs = GridSearchCV(base_model, grid, cv=5, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
    
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best CV F1-Score: {gs.best_score_:.4f}")
    
    return gs.best_estimator_

def generate_confusion_matrix(model, X_test, y_test, title, output_dir="notebooks"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, "best_model_confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix saved to {output_dir}")

def extract_feature_importance(model, best_name, preprocessor_path="models/preprocessor.pkl", orig_columns=None, output_dir="notebooks"):
    print("\n--- Feature Importance Summary ---")
    try:
        preprocessor = joblib.load(preprocessor_path)
        feature_names = preprocessor.get_feature_names_out()
    except Exception as e:
        print(f"Could not load preprocessor features: {e}. Attempting fallback.")
        feature_names = [f"Feature {i}" for i in range(100)]
        
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    
    if importances is None:
        print(f"{best_name} does not explicitly provide feature importances or coefficients.")
        return
        
    feat_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    print("\nTop 10 Most Important Features:")
    print(feat_df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
    plt.title(f'Top 10 Feature Importances ({best_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()
    print("Feature importance chart saved.")

    print("\nMedical Context (General Expectations):")
    print("- 'cp_...': Chest pain type is highly correlated with heart disease.")
    print("- 'thalach': Maximum heart rate achieved during stress.")
    print("- 'exang': Exercise-induced angina.")
    print("- 'oldpeak': ST depression; stress test abnormality.")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, orig_cols = load_and_preprocess_data()
    models, cv_scores = train_base_models(X_train, y_train)
    
    # Evaluate
    results_df, best_name = evaluate_models(models, X_test, y_test)
    
    # Tune Best Model
    tuned_model = tune_best_model(models, best_name, X_train, y_train)
    
    # Save Best Tuned Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(tuned_model, "models/best_model.pkl")
    print("\nSaved Best Tuned Model to 'models/best_model.pkl'")
    
    # Plot Confusion Matrix
    generate_confusion_matrix(tuned_model, X_test, y_test, best_name)
    
    # Feature Importance
    extract_feature_importance(tuned_model, best_name, orig_columns=orig_cols)
    print("\nPipeline Evaluation Complete.")
