PulseGuard AI

A machine learning model that classifies a patient's cardiac risk level from basic vital signs and medical history.

PulseGuard AI takes routine clinical measurements (heart rate, blood pressure, oxygen saturation, chest pain severity, etc.) and predicts one of four risk levels: Low, Moderate, High, or Critical. The final model, a tuned Decision Tree, reaches 96.4% accuracy on a held-out test set.

⚠️ Disclaimer: This project is for educational and portfolio purposes only. It is not a medical device and must not be used for real clinical decisions.

📌 Table of Contents
Problem Statement
Dataset
Project Workflow
Models & Results
Tech Stack
Getting Started
Using the Saved Model
Limitations & Future Work
Author
🎯 Problem Statement

Triaging patients quickly matters in cardiac care. The goal of this project is to build a multi-class classifier that estimates a patient's cardiac risk level from a small set of easily collected features, and to compare a simple baseline (Logistic Regression) against a tree-based model.

📊 Dataset
File: Heart_risk_dataset.csv
Size: 22,500 records, 11 columns (10 after dropping the patient ID)
Missing values: none
Duplicates: none
Feature	Type	Description
age	Numerical	Patient age (18–92)
gender	Categorical	Male / Female
smoker_status	Categorical	Yes / No
diabetes_history	Categorical	Yes / No
heart_rate	Numerical	Beats per minute
systolic_blood_pressure	Numerical	mmHg
oxygen_saturation	Numerical	SpO₂ (%)
chest_pain_severity	Numerical	Scale from 0 to 10
body_temperature	Numerical	°C
risk_level	Target	Low / Moderate / High / Critical Risk

Class distribution (approx.): Low Risk 39.7% · Moderate Risk 27.9% · Critical Risk 18.5% · High Risk 13.9%. The classes are imbalanced, so class_weight='balanced' is used for both models.

🔄 Project Workflow
Data loading & inspection: shape, dtypes, summary statistics.
Data cleaning: checked for nulls and duplicates; dropped patient_id.
Exploratory Data Analysis (EDA)
Target class distribution
Categorical feature distributions
Risk level vs. diabetes history, smoker status, and gender
Correlation heatmap (using an ordinal encoding of the risk level)
Box plots of risk level vs. oxygen saturation, chest pain severity, and heart rate
Preprocessing: a ColumnTransformer with StandardScaler for numerical features and OneHotEncoder for categorical features, wrapped in a scikit-learn Pipeline so the same transformations are applied at inference time.
Train/test split: 80% / 20% (random_state=42).
Modeling: Logistic Regression (baseline), Decision Tree, and a tuned Decision Tree.
Hyperparameter tuning: GridSearchCV with 5-fold cross-validation, optimizing macro F1.
Model export: the best pipeline is saved with joblib as model.pkl.
🏆 Models & Results

Evaluated on the 4,500-sample test set:

Model	Accuracy	Weighted F1
Logistic Regression (baseline)	86.6%	0.867
Decision Tree (max_depth=5)	89.5%	0.897
Decision Tree (tuned with GridSearchCV) ✅	96.4%	0.964

Best hyperparameters:

python
{
    "criterion": "entropy",
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

Best cross-validated macro F1: 0.954

Per-class performance (final model)
Class	Precision	Recall	F1-score
Low Risk	0.99	0.98	0.98
Moderate Risk	0.95	0.95	0.95
High Risk	0.90	0.95	0.92
Critical Risk	0.98	0.96	0.97

Most remaining errors are between neighboring risk levels (e.g., High ↔ Critical, Moderate ↔ High), and the model rarely confuses Low Risk with the higher-risk classes.

🛠️ Tech Stack
Language: Python 3
Data & analysis: pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine learning: scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, DecisionTreeClassifier, GridSearchCV)
Model persistence: joblib
🚀 Getting Started
1. Clone the repository
bash
git clone https://github.com/<your-username>/PulseGuard-AI.git
cd PulseGuard-AI
2. Install dependencies
bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib jupyter
3. Add the dataset

Place Heart_risk_dataset.csv in the project folder and update the path in the data-loading cell if needed (the notebook was written in Google Colab and reads from /content/).

4. Run the notebook
bash
jupyter notebook PulseGuard_Ai.ipynb
🔮 Using the Saved Model

The saved pipeline already includes all preprocessing, so you can pass raw feature values directly:

python
import joblib
import pandas as pd

model = joblib.load("model.pkl")

patient = pd.DataFrame([{
    "age": 54,
    "gender": "Female",
    "smoker_status": "Yes",
    "diabetes_history": "Yes",
    "heart_rate": 131,
    "systolic_blood_pressure": 179,
    "oxygen_saturation": 82.8,
    "chest_pain_severity": 7.8,
    "body_temperature": 37.5,
}])

print(model.predict(patient)[0])  # e.g. "Critical Risk"
📁 Project Structure
PulseGuard-AI/
├── PulseGuard_Ai.ipynb        # Full workflow: EDA, modeling, tuning, export
├── Heart_risk_dataset.csv     # Dataset
├── model.pkl                  # Trained pipeline (tuned Decision Tree)
└── README.md
⚠️ Limitations & Future Work
Mild overfitting: the tuned tree scores 99.0% accuracy on training data vs. 96.4% on test data. Cross-validated macro F1 (0.954) is consistent with the test result, but more regularization or ensemble methods could narrow the gap.
Single split evaluation: final metrics come from one 80/20 split; repeated or stratified cross-validation would give a more robust estimate.
Stratification: train_test_split is not stratified, which could be improved given the class imbalance.
Model comparison: only Logistic Regression and Decision Trees were tested. Random Forest, Gradient Boosting, or XGBoost are natural next steps.
Explainability: adding feature-importance analysis or SHAP values would make predictions easier to interpret.
Deployment: wrap the model in a simple web app (Streamlit or FastAPI) for interactive predictions.
Data provenance: results depend entirely on this dataset's quality and labeling; validation on real clinical data would be required before any real-world use.

👩‍💻 Author

Hadeer Mohammed Built as a project for the NTI Machine Learning program.
