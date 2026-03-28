import streamlit as st
import pandas as pd
from PIL import Image
import os
import joblib
import numpy as np

# Set Page Config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #dc3545;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #c82333;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("❤️ Heart Disease Detection System")
st.markdown("Predict the likelihood of heart disease accurately using Machine Learning.")
st.markdown("---")

# Sidebar - Input Features
st.sidebar.header("🩺 Enter Patient Details")

def user_input_features():
    age = st.sidebar.number_input("Age (Years)", 0, 120, 50)
    sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1], index=1)
    cp = st.sidebar.selectbox("Chest Pain Type (cp) [0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic]", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy)", [0, 1, 2])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversable Defect)", [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return data

patient_data = user_input_features()

# Main Container
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Selected Patient Summary")
    st.dataframe(pd.DataFrame([patient_data]).T.rename(columns={0: 'Value'}))
    
    # Predict button
    if st.button("🔍 Run Prediction"):
        try:
            from src.predict import predict_heart_disease
            from src.data_preprocessing import load_and_preprocess_data
            
            # Use the existing function
            result_str = predict_heart_disease(patient_data)
            
            # Prettify the visual
            st.markdown("### 📊 Prediction Result")
            if "Output: 1" in result_str:
                st.error(f"{result_str}", icon="⚠️")
                conf_str = result_str.split("Confidence: ")[1].split("%")[0]
                conf_val = min(1.0, float(conf_str)/100.0)
                st.progress(conf_val, text="Confidence Level")
                st.info("ℹ️ **Result Explained:** The output is **1**, which means the algorithm has detected patterns indicating **Heart Disease**.")
            elif "Output: 0" in result_str:
                st.success(f"{result_str}", icon="✅")
                conf_str = result_str.split("Confidence: ")[1].split("%")[0]
                conf_val = min(1.0, float(conf_str)/100.0)
                st.progress(conf_val, text="Confidence Level")
                st.info("ℹ️ **Result Explained:** The output is **0**, which means the algorithm has evaluated **No Heart Disease**.")
            else:
                st.warning(result_str)
                
        except Exception as e:
            st.error("Model could not be loaded. Please ensure you have run 'python src/evaluation.py' first.")

with col2:
    st.subheader("📈 About the Model & Features")
    with st.expander("ℹ️ Model Information", expanded=True):
        st.write("""
        This prediction engine is built using an ensemble Machine Learning classifier, hyper-tuned via Grid Search. 
        It processes patient metrics using SMOTE balancing, One-Hot Encoding, and Standard Scaling.
        """)
        
    with st.expander("🖼️ Feature Importance Chart"):
        chart_path = "notebooks/feature_importance.png"
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Top 10 Most Critical Heart Disease Indicators")
        else:
            st.write("Feature importance chart not generated yet. Run 'python src/evaluation.py'.")
            
    with st.expander("📉 Model Performance Matrix"):
        cm_path = "notebooks/best_model_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix of the Best Classifier")
        else:
            st.write("Accuracy matrices not generated yet.")

st.markdown("---")
st.markdown("### Project By ASHISH KUMAR | UCI Cleveland Dataset")
