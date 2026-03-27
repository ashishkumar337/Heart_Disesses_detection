from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Heart Disease Detection Project Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 11)
        # Replacing unicode dashes/apostrophes just in case
        text = text.replace('—', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        self.multi_cell(0, 8, text)
        self.ln()

workflow_text = """Project ek linear data-science pipeline follow karta hai. Iska matlab hai ki hum raw-data ko final application tak stepwise laate hain:
1. Data Loading & EDA: Raw CSV file load hoti hai aur graphs create hote hain.
2. Preprocessing (Data Cleaning): Khaali values bharna, ages ko scale karna aur classes (male/female) ko 0,1 me tabdeel karna. SMOTE technique lagayi jati hai taki data (Disease/No-Disease) balance rahe.
3. Model Training: 5 alag-alag algorithms train kiye jate hain (Logistic Regression, Random Forest, XGBoost, Decision Tree, SVM).
4. Evaluation & Tuning: Sabse best model chuna jata hai F1-Score ke dum par aur usko GridSearchCV se fine-tune karke 'best_model.pkl' save kiya jata hai.
5. Prediction UI (Streamlit): Ek Frontend Web UI banaya jata hai jahan users inputs daal ke disease predict kar sakte hain."""

models_text = """Is project me 5 alag Machine Learning Models use hue hain:
- Logistic Regression: Yeh simple aur fast model hai, probability deta hai ki beemari hai ya nahi.
- Random Forest: Ye trees ka jungle / ensemble hota hai. Outliers se easily bacha leta hai.
- XGBoost: Ye extreme gradient boosting hai. Jab data me complex patterns hote hain to ye sabse best accuracy deta hai!
- SVM (Support Vector Machine): High dimensional data separator ke tor par.
- Decision Tree: Basic rule-based AI.

Kyu Use Kiye Gaye? -> Humne in sabhi models me competition karwaya (Cross Validation use karke) taaki jo model un-seen tabular data par best ban ke ubhre (jaise Logistic Regression ya XGBoost) sirf usey final Streamlit engine ke liye chuna jaa sake!"""

dataset_text = """Dataset: UCI Cleveland Heart Disease (Kaggle)
Isme 303 patients ka medical data hai jisme 14 columns hain:
1. age: Umar saalon me
2. sex: 1=Male, 0=Female
3. cp (Chest Pain Type): 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
4. trestbps: Aaram ke waqt Blood Pressure (mm Hg)
5. chol: Khoon me Cholesterol ka level (mg/dl) - 200 se badhkar khatarnaak
6. fbs: Fasting Blood Sugar agar 120 mg/dl se zyada hai to diabetes (1=Yes, 0=No)
7. restecg: Resting ECG (Dil ki dharkhan report). 0=Normal, 1=Abnormal, 2=High Alert!
8. thalach: Maximum heart rate (Dil kitni tezi se dhadka maximum)
9. exang: Kya exercise ya chalne ke baad seene me dard hota hai? (1=Yes, 0=No)
10. oldpeak: ST depression (ECG me graph ka niche girna)
11. slope: Peak exercise ke waqt ST segment ka jhukaw.
12. ca: Kitne major blood vessels blocked hain (0 se 3 tak ki scale).
13. thal: Thalassemia khoon ki problem (1=Normal, 2=Fixed, 3=Reversable/Dangerous).
14. target (Final): 1 = Heart Disease Detected (Beemar), 0 = No Disease (Healthy)."""

pdf = PDF()
pdf.add_page()
pdf.chapter_title('1. Project Step-by-Step Workflow (Hinglish)')
pdf.chapter_body(workflow_text)
pdf.chapter_title('2. Kis Model Ka Use Hua Aur Kyun? (Models Explained)')
pdf.chapter_body(models_text)
pdf.add_page()
pdf.chapter_title('3. Dataset Ki Detailed Jankari (14 Features)')
pdf.chapter_body(dataset_text)
pdf.output('Heart_Disease_Project_Report.pdf')
print("PDF Generated successfully as Heart_Disease_Project_Report.pdf")
