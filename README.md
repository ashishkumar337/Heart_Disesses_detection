# ❤️ Heart Disease Detection System

A complete, end-to-end Machine Learning web application to predict the likelihood of heart disease in patients based on the UCI Cleveland dataset.

## 📂 Project Structure
```text
heart_disease_detection/
├── data/
│   └── heart-disease.csv          # Raw Dataset
├── notebooks/
│   └── EDA.py                     # Exploratory Data Analysis
│   └── *.png                      # Generated EDA & Evaluation plots
├── models/
│   ├── best_model.pkl             # Trained model artifact
│   └── preprocessor.pkl           # Trained data transformation pipeline
├── src/
│   ├── data_preprocessing.py      # Imputation, SMOTE, Scaling logic
│   ├── model_training.py          # 5-fold CV across 5 different classifiers
│   ├── evaluation.py              # Performance metrics & GridSearch Tuning
│   └── predict.py                 # Prediction endpoint logic
├── app.py                         # Streamlit UI
├── requirements.txt               # Dependencies
└── README.md
```

## 📊 Dataset Description
We use the **Kaggle Heart Disease Dataset (UCI Cleveland)** containing 303 patient records and 14 attributes:
- `age`: patient age
- `sex`: 1=Male, 0=Female
- `cp`: Chest Pain type (0-3)
- `trestbps`: Resting Blood Pressure
- `chol`: Serum Cholesterol in mg/dl
- `fbs`: Fasting Blood Sugar > 120 mg/dl
- `restecg`: Resting ECG results
- `thalach`: Maximum Heart Rate achieved
- `exang`: Exercise-induced Angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels (0-3) colored by flourosopy
- `thal`: Thalassemia (0-3 scale)
- `target`: Diagnosis (0=No Disease, 1=Disease)

## ⚙️ Setup & Installation

**1. Clone the repository and navigate inside:**
```bash
cd heart_disease_detection
```

**2. Install all required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Execution Workflow:**
Generate EDA and insights:
```bash
python notebooks/EDA.py
```
Train, Tune, and Evaluate the Models (this saves the `.pkl` files):
```bash
python src/evaluation.py
```
*Models used: Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine.*

**4. Start the Web App:**
```bash
streamlit run app.py
```

## 📈 Model Performance
We evaluated multiple classifiers. The best performing model (e.g. Random Forest / XGBoost / Logistic Regression) was selected based on the highest F1-Score & ROC-AUC performance. We utilize `SMOTE` over-sampling for extreme accuracy on borderline cases and apply `GridSearchCV` to locate optimal hyperparameters.
