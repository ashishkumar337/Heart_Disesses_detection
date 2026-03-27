import joblib
import pandas as pd
import numpy as np

def predict_heart_disease(patient_data: dict) -> str:
    """
    Predicts heart disease based on input patient details.
    
    Args:
        patient_data (dict): Dictionary comprising single patient's data matching dataset columns.
        
    Returns:
        str: Prediction result string formatted with confidence percentage.
    """
    try:
        # Load preprocessor and model cache
        preprocessor = joblib.load("models/preprocessor.pkl")
        model = joblib.load("models/best_model.pkl")
    except Exception as e:
        return f"Error loading models. Have you trained them yet? Detail: {e}"
        
    # Convert input to dataframe
    df = pd.DataFrame([patient_data])
    
    try:
        # Preprocess features
        X_processed = preprocessor.transform(df)
        
        # Predict class
        pred = model.predict(X_processed)[0]
        
        # Predict probability if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_processed)[0]
            confidence = probs[pred] * 100
        else:
            confidence = 100.0  # fallback for models without proba
        if pred == 1:
            return f"**Output: 1** ⚠️ (Heart Disease Detected) | Confidence: {confidence:.1f}%"
        else:
            return f"**Output: 0** ✅ (No Heart Disease Detected) | Confidence: {confidence:.1f}%"

    except Exception as e:
        return f"Prediction Error: Ensure input matches required features. Details: {e}"

if __name__ == "__main__":
    # Test stub
    sample_data = {
         'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233, 'fbs': 1,
         'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 0,
         'ca': 0, 'thal': 1
    }
    print("Testing Prediction Function...")
    print(predict_heart_disease(sample_data))
